import os
from fastai import *
from fastai.vision import *
import pyperclip
import shutil
from rNCallbacks import *

defaults.device = torch.device('cpu')

learn = load_learner('.', 'export_rN_expanded.pkl')

path = Path('./predictCaptures')

# pyperclip.copy(os.path.abspath('captures'))
printMessage = True
nview = 160
maxImages = 20
numImages = 0
newNumImages = 0
reset = 'False'
pyperclip.copy('13 0 0 0')
vcand = np.load('vcand_case3.npy') # View candidates for case 160 view capture setup

# Load and process view order dictionary.
viewOrd = pickle.load(open("viewOrderDict.pkl", "rb"))
for key, val in viewOrd.items():
    values, counts = np.unique(val, return_counts=True)
    viewOrd[key] = values[np.argmax(counts)]
print(viewOrd)

# Clear directory of any images
for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    try:
        shutil.rmtree(filepath)
    except OSError:
        os.remove(filepath)

try:

    while True:

        while True:
            reset = pyperclip.paste()
            xb = torch.FloatTensor()
            # User wants to reset or max number of views reached, so delete all images and start over
            if numImages == maxImages or reset == 'True':
                print('\nRESETTING...')
                print('Final prediction:', values[np.argmax(counts)])
                numImages = 0
                newNumImages = 0
                pyperclip.copy('13 0 0 0')
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    try:
                        shutil.rmtree(filepath)
                    except OSError:
                        os.remove(filepath)
                        
            # If a new image has been captured, add it to the batch
            try:
                # Wait until a new image is captured before continuing
                while newNumImages == numImages:
                    if printMessage:
                        print('\n***************************')
                        print('Move around the object to scan...\n')
                        printMessage = False
                    newNumImages = len(os.listdir(path))
                    reset = pyperclip.paste()
                    if reset == 'True':
                        break
                numImages = newNumImages
                printMessage = True

                if os.listdir(path):
                    # print('Scanning...')
                    for fname in os.listdir(path):
                        # print('Data detected. Evaluating...', fname)
                        img = open_image(path/fname)
                        # print('Adding data...', fname)
                        batch = learn.data.one_item(img)
                        b,_ = batch
                        xb = torch.cat((xb, b), dim=0)
                        # print('Data sucessfully added!')

                # for i in range(len(os.listdir(path))):
                #     fname = 'view_'+ str(i+1) + '.png'
                #     if i == 0:
                #         print('Scanning...')
                #     img = open_image(path/fname)
                #     print('Adding data...')
                #     batch = learn.data.one_item(img)
                #     b,_ = batch
                #     xb = torch.cat((xb, b), dim=0)
                #     print('Data sucessfully added!')
            
            except OSError:
                print('Warning: Data is unuseable. Continue scanning object...')
                printMessage = True
                break
                               
            xb = [xb]
            learn.model.eval()
            out = learn.model(*xb)
            batch_size = xb[0].size(0)
            num_classes = int(out.size( 1 )/ nview) - 1
            out = out.view( -1, num_classes + 1 )
            out = torch.nn.functional.log_softmax( out, dim=1 )
            out = out[ :, :-1 ] - torch.t( out[ :, -1 ].repeat(1, out.size(1)-1).view(out.size(1)-1, -1) )
            predictions = []
            posePreds = []
            for i in range(batch_size): 
                maxVals, idxs = torch.max(out[i*nview:i*nview+nview], dim=1)
                predictions.append(idxs[torch.argmax(maxVals)].item())
                # posePreds.append(torch.argmax(maxVals).item())
                posePred = torch.argmax(maxVals).item()
                # print(predictions, posePreds)
            values, counts = np.unique(predictions, return_counts=True)
            if predictions[-1] == values[np.argmax(counts)]:
                pose = vcand[viewOrd[predictions[-1]]][posePred]
                if pose < 80:
                    rotation = str( '0 ' + str(-16*(pose//16)-16) + ' ' + str(-22.5*(pose-((pose//16)*16))) )
                else:
                    pose = pose - 80
                    rotation = str( '0 ' + str(16*(pose//16)+16) + ' ' + str(-22.5*(pose-((pose//16)*16))) )
                    pose = pose + 80
                preds = str(values[np.argmax(counts)]) + ' ' + str(rotation)
                print('Predictions: ', preds)
                print(pose)
                pyperclip.copy(preds)
            else:
                print('\nAmbiguous view. Continue scanning object...')
                print('Current evaluation', predictions[-1], 'does not match best prediction.')

                            

                
except (KeyboardInterrupt, SystemExit):
    print('Exiting...')

        