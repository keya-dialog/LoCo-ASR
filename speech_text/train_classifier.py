import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from slurp_embeddings_and_targets import SLURPEmbeddingsTargets
from intent_classifier import IntentClassifier 

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train an Intent Classifier with SpeechT5 embeddings from SLURP dataset')
parser.add_argument('--modality', '-m', choices=['text', 'audio'], default='text', required=True, help='Modality (text or audio).')
parser.add_argument('--pooling', '-p', choices=['average', 'max', 'attention'], default='average', required=True, help='Pooling strategy (average,max,attention)')
parser.add_argument('--version', '-v', choices=['fine_tuned', 'base'], default='fine_tuned', required=True, help='Choose the version of the model (fine-tuned, base)')
args = parser.parse_args()
modality = args.modality
pooling = args.pooling
model_version = args.version

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

if model_version == "fine_tuned":
    folder = "extracted/speecht5"
else:
    folder = "extracted/speecht5_base"

train_original_set = SLURPEmbeddingsTargets(folder, modality, "train")
train_synthetic_set = SLURPEmbeddingsTargets(folder, modality, "train_synthetic")
train_set = train_original_set + train_synthetic_set
val_set = SLURPEmbeddingsTargets(folder, modality, "devel")
test_set = SLURPEmbeddingsTargets(folder, modality, "test")

#train_size = int(0.8 * len(training_set))
#test_size = len(training_set) - train_size
#train_set, val_set = torch.utils.data.random_split(training_set, [train_size, test_size], generator=torch.Generator().manual_seed(42))
print(f"Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}")

#slurp_id, embedding, target = test_set[0]
#print(f"ID: {slurp_id}, Embedding: {embedding.shape}, Target: {target}")

def collate_fn(batch):
    slurp_ids, embeddings, targets = zip(*batch)
    embeddings = pad_sequence(embeddings, batch_first=True)
    targets = torch.stack(targets, dim=0)
    return slurp_ids, embeddings, targets

batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = IntentClassifier(method=pooling, embedding_size=768)
model = model.to(device)

num_epochs = 100
learning_rate = 0.001
patience = 5 #For early stopping
print_every = 200

criterion = nn.CrossEntropyLoss()
criterion_val = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

model_folder = os.path.join("checkpoints",model_version)
save_folder = os.path.join(os.path.join(model_folder, modality), pooling)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

results_folder = os.path.join(os.path.join(os.path.join("results", model_version), modality), pooling)
plots_folder = os.path.join(results_folder, "plots")
logs_folder = os.path.join(results_folder, "logs")
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)

def train(model, train_loader, val_loader, criterion, criterion_val, optimizer, num_epochs, patience=5, print_every=200):
    text_to_write = "Results\n"
    
    total_loss = []
    val_loss_list = []
    acc_list = []
    acc_val_list = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        acc_train = 0.0
        model.train()
        
        for i, (_, data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(data)
            
            # Compute the loss
            pred = pred.squeeze(1)

            loss = criterion(pred, target.float())

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            acc = float((torch.argmax(pred, 1) == torch.argmax(target, 1)).float().sum())
            acc_train += acc
            
            # Print the loss every specified number of iterations
            if (i + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                text_to_write += f"Epoch [{epoch+1}/{num_epochs}], Iteration [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}\n"
                
        epoch_loss /= len(train_loader)
        total_loss.append(epoch_loss)
        acc_train /= len(train_set)
        #acc_train = round(acc_train, 2)
        acc_list.append(acc_train)
        torch.save(model.state_dict(), os.path.join(save_folder, f'speecht5_{pooling}_{modality}_epoch_{epoch+1}.pth'))
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Compute the validation loss
            val_loss = 0.0
            acc_val = 0.0
            for _, data_eval, target_eval in val_loader:
                data_eval = data_eval.to(device)
                target_eval = target_eval.to(device)
                pred_eval = model(data_eval)
                pred_eval = pred_eval.squeeze(1)
                
                val_loss += criterion_val(pred_eval, target_eval.float()).item()
                acc_val += float((torch.argmax(pred_eval, 1) == torch.argmax(target_eval, 1)).float().sum()) #mean and remove round *100
            val_loss /= len(val_set)
            val_loss_list.append(val_loss)
            acc_val /= len(val_set)
            #acc_val = round(acc_val,2)
            acc_val_list.append(acc_val)

        # Print the epoch loss and validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training accuracy: {round(acc_train*100,2)}, Validation Loss: {val_loss:.4f}, Validation accuracy: {acc_val*100:.2f}")
        text_to_write += f"###### Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training accuracy: {round(acc_train*100,2)}, Validation Loss: {val_loss:.4f}, Validation accuracy: {acc_val*100:.2f} ######\n\n"

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model weights
            torch.save(model.state_dict(), os.path.join(save_folder, f'speecht5_{pooling}_{modality}_best.pth'))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: Validation loss has not improved in the last {} epochs.".format(patience))
            break

    torch.save(model.state_dict(), os.path.join(save_folder, f'speecht5_{pooling}_{modality}_last.pth'))

    with open(os.path.join(logs_folder, "results.txt"), 'w') as logss:
        logss.write(text_to_write)

    # Plot the loss function over iterations
    plt.figure()
    plt.plot(total_loss, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    #plt.show(block=True)
    plt.savefig(os.path.join(plots_folder, "losses.png"))

    # Plot the accuracy metric over iterations
    plt.figure()
    plt.plot(acc_list, label='Training Accuracy')
    plt.plot(acc_val_list, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    #plt.show(block=True)
    plt.savefig(os.path.join(plots_folder, "accuracies.png"))

def evaluate(model, test_loader, criterion_val):
    model.eval()
    test_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for _, data_test, target_test in test_loader:
            data_test = data_test.to(device)
            target_test = target_test.to(device)
            pred_test = model(data_test)
            pred_test = pred_test.squeeze(1)
            test_loss += criterion_val(pred_test, target_test.float()).item()
            acc += float((torch.argmax(pred_test, 1) == torch.argmax(target_test, 1)).float().sum())

        test_loss /= len(test_set)
        acc /= len(test_set)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {acc*100:.2f}")

print("Training started...")
train(model, train_loader, val_loader, criterion, criterion_val, optimizer, num_epochs, patience=patience, print_every=print_every)
print("Training done!")

model = IntentClassifier(method=pooling, embedding_size=768).to(device)
model.load_state_dict(torch.load(os.path.join(save_folder, f'speecht5_{pooling}_{modality}_best.pth')))

print("Evaluating model on test set")
evaluate(model, test_loader, criterion_val)
print("Evaluation done!")