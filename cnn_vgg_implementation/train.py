import torch
import os
import data_setup,engine,model_builder,utils

from torchvision import transforms

NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

train_dir='data/pizza_steak_sushi/train'
test_dir='data/pizza_steak_sushi/test'
if __name__ == '__main__':
#setup target dveice
    device="cuda" if torch.cuda.is_available() else "cpu"

    #transform
    data_transform=transforms.Compose([transforms.Resize((64,64)),
                                    transforms.ToTensor()])
    #create dataloader
    train_dataloader,test_dataloader,class_names=data_setup.create_dataloaders(
        train_dir,test_dir,data_transform,BATCH_SIZE
    )

    #create model w model_builder.py

    model=model_builder.TinyVGG(3,HIDDEN_UNITS,len(class_names)).to(device)

    #setup loss and optim
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    #start training
    engine.train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,loss_fn=loss_fn,optimizer=optimizer,epochs=NUM_EPOCHS,device=device)

    utils.save_model(model=model,target_dir="models",model_name="05_going_modular_script_mode_mobilenet_model.pth")
