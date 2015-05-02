--require('mobdebug').start()
--------------------------------------------------------------------
require 'nn'
require 'optim'
require 'xlua'
require 'DataSet'

sgd_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-4,
  weightDecay = 1e-4,
  momentum = 0.9
}
opt={
  epochs = 100,
  batch_size = 512,
}

----------------------------------------------------------------------
-- 1. Create the training data

print('')
print('============================================================')
print('Constructing dataset')
print('')

require 'csvigo'
string_to_class={};
for i=1,9 do
  string_to_class["Class_"..i]=i
end

function load_data(fname)

  loaded = csvigo.load(fname)
  if fname~='test.csv' then
    loaded_targets=loaded.target
    for i=1,#loaded_targets do
      loaded_targets[i]=string_to_class[loaded_targets[i]]
    end

    targets = torch.Tensor(loaded_targets)
  end
  local mock=torch.Tensor(loaded['feat_1'])
  local size=(#mock)[1]
  inputs = torch.Tensor( size,93 )
  
  for i=2,94 do
    inputs[{ {},i-1 }] = torch.Tensor(loaded["feat_"..(i-1)]);
  end
  shuffle=torch.randperm(size)
  shuffle_inputs=torch.Tensor(size,93 )
  shuffle_targets=torch.Tensor(size)
  for i=1,size do
    shuffle_inputs[i]=inputs[shuffle[i]];
    if targets then
      shuffle_targets[i]=targets[shuffle[i]];
    end
  end
  
  if fname=='train.csv' then
      local k=math.floor(size/3);
      valid_inputs=shuffle_inputs[{{1,k}}]
      valid_targets=shuffle_targets[{{1,k}}]
      train_inputs=shuffle_inputs[{{k+1,size}}]
      train_targets=shuffle_targets[{{k+1,size}}]
      return DataSet(train_inputs,train_targets,opt),DataSet(valid_inputs,valid_targets,opt)
  end
  return DataSet(shuffle_inputs,shuffle_targets,opt)
end

train_dataset,valid_dataset=load_data('train.csv')
--test_dataset=load_data('test.csv')

print('train size: '..train_dataset:size())
print('valid size: '..valid_dataset:size())
criterion = nn.ClassNLLCriterion()

model = nn.Sequential()
model:add(nn.Linear(93,9))
model:add(nn.LogSoftMax())

x, dl_dx = model:getParameters()



-- classes
classes = {}

for i=1,93 do
  table.insert(classes,''..i)
end
-- This matrix records the current confusion across classes
train_confusion = optim.ConfusionMatrix(classes)
valid_confusion = optim.ConfusionMatrix(classes)

for i = 1,opt.epochs do

  train_confusion:zero()
  valid_confusion:zero()
  current_loss = 0
  xlua.progress(i,opt.epochs)

  for batch = 1,math.ceil(train_dataset:size()/opt.batch_size) do
    inputs,targets=train_dataset:getBatch(batch)
    local feval = function(x_new)
      dl_dx:zero()
      output = model:forward(inputs)

      train_confusion:add(output, targets[i])

      local loss_x = criterion:forward(output, targets)
      model:backward(inputs, criterion:backward(loss_x, targets))

      return loss_x, dl_dx
    end

    _,fs = optim.sgd(feval,x,sgd_params)

    current_loss = current_loss + fs[1]
  end

  -- Validation
  for batch = 1,math.ceil(valid_dataset:size()/opt.batch_size) do

    _,inputs,targets=valid_dataset.getBatch(batch)
    output = model:forward(inputs)

    valid_confusion:add(output, targets[i])
  end

  -- report average error on epoch
  current_loss = current_loss / (#dataset_inputs)[1]
  print('epoch = ' .. i .. 
    ' of ' .. epochs .. 
    ' current loss = ' .. current_loss)
  print('Training accuracy:')
  print(train_confusion)
  print('Validation accuracy:')
  print(valid_confusion)
end



lbfgs_params = {
  lineSearch = optim.lswolfe,
  maxIter = epochs,
  verbose = true
}



