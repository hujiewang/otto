--require('mobdebug').start()
--------------------------------------------------------------------
require 'nn'
require 'optim'
require 'xlua'
require 'DataSet'
require 'image'
require 'predict'
require 'load_data'
require 'tools'
require 'gnuplot'


optim_func=optim.rmsprop
optim_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-6,
  weightDecay = 1e-4,
  dampening = 0.3,
  momentum = 0.95,
  nesterov,
}
opt={
  createData = false,
  epochs = 2000,
  batch_size = 10000,
  predict = false,
  save_gap = 10,
  cuda=true,
  plot=false,
  sparse_init = true,
  standardize = true,
  --model_file = 'model.dat'
  RF = false,
}

print(opt)
----------------------------------------------------------------------
math.randomseed( os.time() )
--[[GPU or CPU]]--
if opt.cuda then
  require 'cutorch'
  require 'cunn'
  --torch.setdefaulttensortype('torch.CudaTensor')
  print('Global: switching to CUDA')
end
require 'model'
require 'csvigo'
string_to_class={};
for i=1,9 do
  string_to_class["Class_"..i]=i
end

if opt.predict then
  predict()
  return
end

print('')
print('============================================================')
print('Constructing dataset')
print('')

if opt.createData then
  train_dataset,valid_dataset=load_data('train.csv')
  torch.save('./data/data.dat',{['train']=train_dataset,['valid']=valid_dataset})
  print('Data has been saved!, now exiting...')
  return
else
  data=torch.load('./data/data.dat')
  train_dataset=data['train']
  valid_dataset=data['valid']
  train_dataset.opt=opt
  valid_dataset.opt=opt
  print('Data has been loaded!')
end

if opt.standardize then
  print('Standardizing dataset...')
  standardize(train_dataset,valid_dataset)
end
print('train size: '..train_dataset:size())
print('valid size: '..valid_dataset:size())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------


if opt.model_file then
  print('Loading existing model...')
  model_data=torch.load("./model/model.dat")
  model=model_data['model']
end

--[[GPU or CPU]]--
if opt.cuda then
  model:cuda()
  criterion:cuda()
  if train_dataset then
    train_dataset:type('cuda')
  end
  if valid_dataset then
    valid_dataset:type('cuda')
  end
end

if opt.RF then
    train_dataset:RandomPatch(math.ceil(train_dataset:size()/2))
    print('RF new train size: '..train_dataset:size())
end
if opt.sparse_init then
  print('Using sparse init')
  sparseReset(model)
end


x, dl_dx = model:getParameters()

-- classes
classes = {}

for i=1,9 do
  table.insert(classes,''..i)
end
-- This matrix records the current confusion across classes
train_confusion = optim.ConfusionMatrix(classes)
valid_confusion = optim.ConfusionMatrix(classes)

os.execute("rm ./log/logger.log")
logger = optim.Logger("./log/logger.log")
logger:setNames({'# train', 'valid'})

best_validation_error = math.huge
last_save = 0
for i = 1,opt.epochs do

  train_confusion:zero()
  valid_confusion:zero()
  train_loss = 0
  valid_loss = 0
  xlua.progress(i,opt.epochs)
  
  -- training 
  model:training()
  for batch = 1,math.ceil(train_dataset:size()/opt.batch_size) do
    --xlua.progress(batch,math.ceil(train_dataset:size()/opt.batch_size))
    inputs,targets=train_dataset:getBatch(batch)
    local feval = function(x_new)
      dl_dx:zero()
      output = model:forward(inputs)
      train_confusion:batchAdd(output, targets)
      
      local loss_x = criterion:forward(output, targets)
      model:backward(inputs, criterion:backward(output, targets))

      return loss_x, dl_dx
    end

    _,fs = optim_func(feval,x,optim_params)

    train_loss = train_loss + fs[1]*(#inputs)[1]
  end
  
  -- Validation
  model:evaluate()
  for batch = 1,math.ceil(valid_dataset:size()/opt.batch_size) do

    inputs,targets=valid_dataset:getBatch(batch)
    output = model:forward(inputs)
    valid_confusion:batchAdd(output, targets)
    local loss_x = criterion:forward(output, targets)
    valid_loss = valid_loss + loss_x*(#inputs)[1]
  end

  train_dataset:shuffleComplete()
  --train_dataset:shuffleData()
  -- report average error on epoch
  train_loss = train_loss / train_dataset:size()
  valid_loss = valid_loss / valid_dataset:size()
  print('epoch = ' .. i .. ' of ' .. opt.epochs)

  train_confusion:updateValids()
  valid_confusion:updateValids()
  print('train loss:')
  print(train_loss)
  print('Training accuracy:')
  print(train_confusion.totalValid * 100)
  print('valid loss:')
  print(valid_loss)
  print('Validation accuracy:')
  print(valid_confusion.totalValid * 100)
  print('best valid loss')
  print(best_validation_error)
  logger:add({train_loss,valid_loss})
  if best_validation_error>valid_loss then
    print('Found new optima with valid error = '..valid_loss)
    if i-last_save >=opt.save_gap then
      last_save=i
      os.execute('rm ./model/model.dat')
      torch.save('./model/model.dat',{['epoch']=i,
                              ['train_accuracy']=train_confusion.totalValid,
                              ['train_loss']=train_loss,
                              ['valid_accuracy']=valid_confusion.totalValid,
                              ['valid_loss']=valid_loss,
                              ['model']=model})
      print('Model has been saved!')
    end
    best_validation_error=valid_loss
  end
  if opt.plot then
    logger:style({'-','-'})
    logger:plot()
  end
end






