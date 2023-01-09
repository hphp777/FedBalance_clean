from torchvision import transforms

NIH_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
ChexPert_normalize = transforms.Normalize(mean=[0.485],
                                 std=[0.229])

# transforms.RandomHorizontalFlip() not used because some disease might be more likely to the present in a specific lung (lelf/rigth)
NIH_transform = transforms.Compose([transforms.ToPILImage(), 
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    NIH_normalize])

ChexPert_transform = transforms.Compose([transforms.Resize([256,256]),
                    transforms.ToTensor(),
                    ChexPert_normalize])