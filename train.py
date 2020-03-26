import traceback
import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from data.stl10 import get_loaders as stl10_loaders
from utils.general import make_directory, TrainingConfig, DatasetConfig
from utils.torch import variational_ELBO, weighted_variational_ELBO, save_training_data, log
from models.vae import VAE
from models.vae_conv import ConvVAE
from models.vae_inception import InceptionVAE

config = TrainingConfig(cuda=True, batch_size=64, log_interval=100, epochs=10, output_dir_name='results')
data_config = DatasetConfig(image_size=96, channels=3)

device = torch.device("cuda" if config.cuda else "cpu")
train_loader, test_loader = stl10_loaders(config.batch_size, shuffle=True, num_workers=0)

# define the network model and the optimizer
# can replace by either: VAE, ConvVAE, or InceptionVAE

model = InceptionVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# log model and optimizer
log(model=model, optimizer=optimizer)


def train(current_epoch):
    model.train()
    train_loss, epoch_loss = 0, []

    for batch_idx, (data_batch, _) in enumerate(train_loader):
        # forward pass through network, get reconstructions
        data_batch = data_batch.to(device)
        recon_batch, mu, log_var = model(data_batch)

        # calculate reconstruction loss
        loss = variational_ELBO(recon_batch, data_batch, mu, log_var)

        # backprop
        # SGD in optimal gradient direction
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        train_loss += loss.item()
        epoch_loss.append(loss.item() / len(data_batch))

        if batch_idx % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(current_epoch, batch_idx * len(data_batch),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item() / len(data_batch)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        current_epoch, train_loss / len(train_loader.dataset)))
    return epoch_loss


def test(current_epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (data_batch, _) in enumerate(test_loader):
            data_batch = data_batch.to(device)
            recon_batch, mu, log_var = model(data_batch)

            # for testing, simply forward propagate
            # and record loss

            test_loss += variational_ELBO(recon_batch, data_batch, mu, log_var).item()

            if i == 0:
                n = min(data_batch.size(0), 8)
                comparison = torch.cat(
                    [data_batch[:n], recon_batch.view(config.batch_size, data_config.channels, data_config.image_size,
                                                      data_config.image_size)[:n]])
                save_image(comparison.cpu(),
                           config.output_dir_name + '/reconstruction_' + str(current_epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def start_training():
    # keep track of loss values
    historic_training_loss = []
    historic_testing_loss = []

    for epoch in range(1, config.epochs + 1):
        epoch_train_loss = train(epoch)
        epoch_test_loss = test(epoch)

        historic_training_loss.extend(epoch_train_loss)
        historic_testing_loss.append(epoch_test_loss)

    save_training_data(historic_training_loss, historic_testing_loss, output_dir=config.output_dir_name)


if __name__ == "__main__":
    make_directory(dir_name=config.output_dir_name)
    try:
        start_training()
    except Exception as e:
        print(traceback.format_exc())
