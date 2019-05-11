from fastai import *
from fastai.vision import *

class GenerateTargetrN(LearnerCallback):

    "Creates the target labels for a rotationNet implementation."
    
    def __init__(self, learn:Learner):
        super().__init__(learn)
        self.target_var = torch.LongTensor( learn.data.train_dl.dl.batch_size * nview )
        self.output = torch.Tensor()
        self.viewOrd = {}
    
    def on_loss_begin(self, last_output, last_target, train, **kwargs:Any)->None:
        
        if not train:
            return {'last_output':last_output, 'last_target':last_target}

        nsamp = int( last_output.size(0) / nview )
        target_ = torch.LongTensor( last_target.size(0) * nview )

        self.output = last_output
        num_classes = int( self.output.size( 1 ) / nview ) - 1
        self.output = self.output.view( -1, num_classes + 1 )

        # compute scores and decide target labels
        output_ = torch.nn.functional.log_softmax( self.output, dim=1 ) #try sigmoid instead?

        #subtracts the last col from every other col, removes last col
        output_ = output_[ :, :-1 ] - torch.t( output_[ :, -1 ].repeat(1, output_.size(1)-1).view(output_.size(1)-1, -1) )
        output_ = output_.view( -1, nview * nview, num_classes )
        output_ = output_.data.cpu().numpy()

        '''Arrange output_as:
           [x,:,:] = all the view's activations for image[x//num views]
           [:,y,:] = view activations for class[y]
           [:,:,z] = sample set[z]'''
        output_ = output_.transpose( 1, 2, 0 )

        #default view variable is the incorrect view
        for j in range(target_.size(0)):
            target_[ j ] = num_classes

        # Initialize scores to 0
        scores = np.zeros( ( vcand.shape[ 0 ], num_classes, nsamp ) )

        #add up scores for each of the candidates for viewpoint variables
        for j in range(vcand.shape[0]):
            for k in range(vcand.shape[1]):
                scores[ j ] = scores[ j ] + output_[ vcand[ j ][ k ] * nview + k ]

        for n in range( nsamp ):
            #finds max score for column that corresponds to target label (for each class label and sample group)
            j_max = np.argmax( scores[ :, last_target[ n * nview ], n ] )
            
            if last_target[n*nview].item() in self.viewOrd:
                self.viewOrd[last_target[n*nview].item()] = np.append(self.viewOrd[last_target[n*nview].item()], [j_max])
                          
#                 if self.viewOrd[last_target[n*nview].item()] != j_max:
#                     self.viewOrd[last_target[n*nview].item()] = j_max
            else:
                self.viewOrd.update( { last_target[n*nview].item() : [j_max] } )
                
            # Assign target labels. Only 1 view per image gets set to class label.
            # Others remain default, which is the incorrect view
            for k in range(vcand.shape[1]):
                target_[ n * nview * nview + vcand[ j_max ][ k ] * nview + k ] = last_target[ n * nview ]

        target_ = target_.cuda()
        self.target_var = torch.autograd.Variable(target_)

        return {'last_output':self.output, 'last_target':self.target_var}

    def on_epoch_end(self, **kwargs):
        
        f = open("viewOrderDict.pkl","wb")
        pickle.dump(self.viewOrd,f)
        f.close()

class rNAccuracy(LearnerCallback):
    _order=-20
    
    "Metrics for rotationNet implementation."
    

    def __init__(self, learn):
        super().__init__(learn)
  
    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['rN_t_loss', 'Prec@1', 'Prec@5'])
        self.vcount_tot = 0
        self.prec1, self.prec5 = 0., 0.
        
    def on_epoch_begin(self, **kwargs):
        self.tloss_tot, self.vloss_tot, self.tcount, self.vcount= 0., 0., 0, 0
        self.incTargs = []
        
        #print weights to confirm body is frozen or not
#         print(list(learn.model.parameters())[0][0][0][1])
        
        # random permutation
        train_nsamp = int( len(learn.data.train_ds) / nview )
        
        inds = np.zeros( ( nview, train_nsamp ) ).astype('int')
        inds[ 0 ] = np.random.permutation(range(train_nsamp)) * nview
        for i in range(1,nview):
            inds[ i ] = inds[ 0 ] + i
        inds = inds.T.reshape( nview * train_nsamp )
        
        IL  = ImageList([sorted_img[i] for i in inds], path=path)
        learn.data.train_ds.x = IL
        CL = CategoryList([sorted_cats[i] for i in inds], classes=classes, path=path)
        learn.data.train_ds.y = CL
        
        # Print a random data item to check if they got shuffled correctly
#         idx = random.randint(0,len(learn.data.train_ds))
#         print(data.train_ds[idx][1], data.train_ds.items[idx])


    def on_batch_end(self, last_loss, last_output, last_target, train, **kwargs):
        
        if train:
            self.tloss_tot += last_loss
            self.tcount += 1
        else:
            self.vloss_tot += last_loss
            self.vcount += 1
            
            # Calculate accuracy
            
            output_ = last_output
            target = last_target
            target = target.cuda()
            target = target[0:-1:nview]
            nsamp = int( output_.size(0) / nview )
            batch_size = target.size(0)
            num_classes = int(output_.size( 1 )/ nview) - 1

            output_ = output_.view( -1, num_classes + 1 )
            output_ = torch.nn.functional.log_softmax( output_, dim=1 ) #try sigmoid instead?
            output_ = output_[ :, :-1 ] - torch.t( output_[ :, -1 ].repeat(1, output_.size(1)-1).view(output_.size(1)-1, -1) )
            output_ = output_.view( -1, nview * nview, num_classes )
            output_ = output_.data.cpu().numpy()
            output_ = output_.transpose( 1, 2, 0 )
            
            scores = np.zeros( ( vcand.shape[ 0 ], num_classes, batch_size ) )
            output = torch.zeros( ( batch_size, num_classes ) )

    
            for j in range(vcand.shape[0]):
                for k in range(vcand.shape[1]):
                    scores[ j ] = scores[ j ] + output_[ vcand[ j ][ k ] * nview + k ]

            for n in range( batch_size ):
                # For a given batch, n, np.argmax( scores[ :, :, n ] ) / scores.shape[ 1 ]
                # gets the index for the view that has the highest score, regardless of class.
                j_max = int( np.argmax( scores[ :, :, n ] ) / scores.shape[ 1 ] )
                # Scores[ j_max, :, n ] -- for batch n, view with highest score regardless of class.
                output[ n ] = torch.FloatTensor( scores[ j_max, :, n ] )

            # output[x,:] - view that had the highest class score for sample[x]
            # output[:,y] - score for class[y]
            output = output.cuda()


            topk = (1,5)
            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

            prec = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                prec.append(correct_k.mul_(100.0 / batch_size))
                
            self.prec1 += prec[0]*(last_output.size(0)//nview)
            self.prec5 += prec[1]*(last_output.size(0)//nview)
            self.vcount_tot += (last_output.size(0)//nview)
            
            # Save off incorrect predictions for evaluation at epoch end.
            idx =  (correct[0] == 0).nonzero()
            self.incTargs = np.append(self.incTargs, target[idx.view(-1)].data.cpu().numpy())

            return {'last_output':output}


    def on_epoch_end(self, last_metrics, epoch, **kwargs):
        
#         global viewOrd
#         viewOrd = self.viewOrd.astype(int)
#         np.save('viewOrder_MIRO.npy', viewOrd)
        
#         print('\n*******EPOCH', epoch, '********\n')
#         print('Error rate by class:')
#         # Count classes that have been incorrectly classified
#         u, counts = np.unique(self.incTargs, return_counts=True)
#         v = [validCount[i] for i in u.astype(int)]
#         unique = [classes[i] for i in u.astype(int)]
#         corrects = ((counts/v * 100) + 0.5).astype(int) / 100.0
#         countDict = dict(zip(unique, corrects))
#         print(countDict, len(countDict))
        

#         print('vcount:', self.vcount)
        return add_metrics(last_metrics, [self.tloss_tot/self.tcount,
                                          self.prec1/self.vcount_tot, self.prec5/self.vcount_tot])