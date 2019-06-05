import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import resnet101
import time
import cPickle
import model as gan_model
from miscc.config import cfg
import imagecaption.opts as opts
from imagecaption.dataloader import *
import imagecaption.misc.utils as utils
from imagecaption.train import add_summary_value

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

class FeatureExtractor(nn.Module):
    def __init__(self, resnet):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet

    def forward(self, img):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        feature = x.mean(3).mean(2).squeeze()

        return feature


# model, training options
#TODO:
opt = opts.parse_opt()

# "decoder": generator of AttnGAN
if cfg.GAN.B_DCGAN:
    model = gan_model.G_DCGAN()
else:
    model = gan_model.G_NET() # text-to-image AttnGAN generator
model.cuda()
model.train()

# resnet feature extractor
resnet = resnet101(pretrained=True)
feature_extractor = FeatureExtractor(resnet)
feature_extractor.cuda()
feature_extractor.eval()

# linear layer to convert dim
transform = nn.Linear(2048, cfg.TEXT.EMBEDDING_DIM, bias=False)

# data loader
loader = DataLoader(opt)
batch_size = loader.batch_size

########## some variables to use with GAN ##########
noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.Z_DIM)).cuda()
word_embs = torch.zeros(batch_size, cfg.TEXT.EMBEDDING_DIM).cuda()
mask = torch.zeros(batch_size, cfg.TEXT.WORDS_NUM).cuda()

def eval_split(model, loader, transform, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        fc_feats = data['fc_feats']
        with torch.no_grad():
            fc_feats = Variable(torch.from_numpy(fc_feats)).cuda()
            transform_embs = transform(fc_feats)
            [fake_imgs], att_maps, mu, logvar = model(noise, transform_embs, word_embs, mask)
            fk_feats, _ = feature_extractor(fake_imgs[-1])
            loss = torch.dist(fc_feats, fk_feats) / loader.batch_size
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    if verbose:
        print('evaluating validation preformance: L2 norm %f' % loss)
    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals

# loggings
tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

infos = {}
histories = {}
if opt.start_from is not None:
    if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')) as f:
            histories = cPickle.load(f)

iteration = infos.get('iter', 0)
epoch = infos.get('epoch', 0)

val_result_history = histories.get('val_result_history', {})
loss_history = histories.get('loss_history', {})
lr_history = histories.get('lr_history', {})

loader.iterators = infos.get('iterators', loader.iterators)
loader.split_ix = infos.get('split_ix', loader.split_ix)
if opt.load_best_score == 1:
    best_val_score = infos.get('best_val_score', None)

update_lr_flag = True

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

# Load the optimizer
if vars(opt).get('start_from', None) is not None:
    optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

# training
while True:
    if update_lr_flag:
        # Assign the learning rate
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
        else:
            opt.current_lr = opt.learning_rate
        update_lr_flag = False

    start = time.time()
    # Load data from train split (0)
    data = loader.get_batch('train')
    print('Read data:', time.time() - start)

    torch.cuda.synchronize()
    start = time.time()

    fc_feats = data['fc_feats']  # 2048
    fc_feats = Variable(torch.from_numpy(fc_feats), requires_grad=False).cuda()

    optimizer.zero_grad()
    transform_embs = transform(fc_feats)
    [fake_imgs], att_maps, mu, logvar = model(noise, transform_embs, word_embs, mask)
    fk_feats = feature_extractor(fake_imgs)
    loss = torch.dist(fc_feats, fk_feats) / batch_size
    loss.backward()
    utils.clip_gradient(optimizer, opt.grad_clip)
    optimizer.step()
    train_loss = loss.item()
    torch.cuda.synchronize()
    end = time.time()
    print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
          .format(iteration, epoch, train_loss, end - start))

    # Update the iteration and epoch
    iteration += 1
    if data['bounds']['wrapped']:
        epoch += 1
        update_lr_flag = True

    # Write the training loss summary
    if (iteration % opt.losses_log_every == 0):
        if tf is not None:
            add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
            tf_summary_writer.flush()

        loss_history[iteration] = train_loss
        lr_history[iteration] = opt.current_lr\

    # make evaluation on validation set, and save model
    if (iteration % opt.save_checkpoint_every == 0):
        # eval model
        eval_kwargs = {'split': 'val',
                       'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))
        val_loss = eval_split(model, loader, transform, eval_kwargs={})

        # Write validation result into summary
        if tf is not None:
            add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
            tf_summary_writer.flush()
        val_result_history[iteration] = {'loss': val_loss}

        # Save model if is improving on validation result
        current_score = - val_loss

        best_flag = False
        if True:  # if true
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'G_DCGAN.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()

            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
                cPickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
                cPickle.dump(histories, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'G_DCGAN-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

    # Stop if reaching max epochs
    if epoch >= opt.max_epochs and opt.max_epochs != -1:
        break
