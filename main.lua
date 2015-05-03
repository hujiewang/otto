--require('mobdebug').start()
--------------------------------------------------------------------
require 'nn'
require 'optim'
require 'xlua'
require 'DataSet'
require 'image'
require 'predict'
require 'load_data'

optim_func=optim.sgd
optim_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-4,
  weightDecay = 1e-4,
  dampening = 0.5,
  momentum = 0.9,
  nesterov,
}
opt={
  createData = false,
  epochs = 100000,
  batch_size = 4096,
  predict = false,
  save_gap = 30,
  cuda=true,
}

print(opt)
----------------------------------------------------------------------
--[[GPU or CPU]]--
if opt.cuda then
  require 'cutorch'
  require 'cunn'
  print('Global: switching to CUDA')
end

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
  torch.save('data.dat',{['train']=train_dataset,['valid']=valid_dataset})
  print('Data has been saved!, now exiting...')
  return
else
  data=torch.load('data.dat')
  train_dataset=data['train']
  valid_dataset=data['valid']
  train_dataset.opt=opt
  valid_dataset.opt=opt
  print('Data has been loaded!')
end

print('train size: '..train_dataset:size())
print('valid size: '..valid_dataset:size())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
model = nn.Sequential()

model:add(nn.Linear(93,512))
model:add(nn.ReLU())
model:add(nn.Dropout())

model:add(nn.Linear(512,256))
model:add(nn.ReLU())
model:add(nn.Dropout())

model:add(nn.Linear(256,128))
model:add(nn.ReLU())
model:add(nn.Dropout())

model:add(nn.Linear(128,64))
model:add(nn.ReLU())
model:add(nn.Dropout())

model:add(nn.Linear(64,32))
model:add(nn.ReLU())
model:add(nn.Dropout())

model:add(nn.Linear(32,9))

model:add(nn.LogSoftMax())

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
  



x, dl_dx = model:getParameters()

-- classes
classes = {}

for i=1,9 do
  table.insert(classes,''..i)
end
-- This matrix records the current confusion across classes
train_confusion = optim.ConfusionMatrix(classes)
valid_confusion = optim.ConfusionMatrix(classes)

best_validation_error = 0.0
last_save = 0
for i = 1,opt.epochs do

  train_confusion:zero()
  valid_confusion:zero()
  current_loss = 0
  xlua.progress(i,opt.epochs)
  
  -- training 
  model:training()
  for batch = 1,math.ceil(train_dataset:size()/opt.batch_size) do
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

    current_loss = current_loss + fs[1]
  end
  
  -- Validation
  model:evaluate()
  for batch = 1,math.ceil(valid_dataset:size()/opt.batch_size) do

    inputs,targets=valid_dataset:getBatch(batch)
    output = model:forward(inputs)

    valid_confusion:batchAdd(output, targets)
  end

  train_dataset:shuffleComplete()
  -- report average error on epoch
  current_loss = current_loss / train_dataset:size()
  print('epoch = ' .. i .. 
    ' of ' .. opt.epochs .. 
    ' current loss = ' .. current_loss)

  train_confusion:updateValids()
  valid_confusion:updateValids()
  print('Training accuracy:')
  print(train_confusion.totalValid * 100)
  print('Validation accuracy:')
  print(valid_confusion.totalValid * 100)
  
  if best_validation_error<valid_confusion.totalValid then
    print('Found new optima with valid accuracy = '..valid_confusion.totalValid)
    if i-last_save >=opt.save_gap then
      last_save=i
      torch.save('model.dat',{['model']=model})
      print('Model has been saved!')
    end
    best_validation_error=valid_confusion.totalValid
  end
end






