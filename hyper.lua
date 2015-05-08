require 'nn'
require 'optim'
require 'xlua'
require 'DataSet'
require 'image'
require 'predict'
require 'load_data'
require 'tools'
require 'gnuplot'

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
  epochs = 2,
  batch_size = 4096,
  predict = false,
  save_gap = 10,
  cuda=true,
  plot=false,
  sparse_init = true,
  standardize = true,
  --model_file = 'model.dat'
}

act={nn.ReLU(),nn.Tanh()}
function randomNet()
  local expert=nn.Sequential()
  local n_layers=math.random(1,4)
  local n_input=93
  local n_output=93
  --expert:add(nn.BatchNormalization(93))
  
  expert:add(nn.Dropout(math.random(5,35)/100.0))

  for i=1,n_layers do
    n_output=math.random(10,600)
    expert:add(nn.Linear(n_input,n_output))
    expert:add(act[math.random(#act)]:clone())
    --expert:add(nn.BatchNormalization(n_output))
    expert:add(nn.Dropout())
    n_input=n_output
  end
  --]]
  expert:add(nn.Linear(n_output,9))
  expert:add(nn.LogSoftMax())
  return expert
end

function getModel()
  local n = 5
  experts = nn.ConcatTable()

  for i = 1, n do
    experts:add(randomNet())
  end

  gater = nn.Sequential()

  gater:add(nn.Linear(93,512))
  gater:add(nn.ReLU())

  gater:add(nn.Linear(512,512))
  gater:add(nn.ReLU())

  gater:add(nn.Linear(512,n))

  gater:add(nn.SoftMax())

  trunk = nn.ConcatTable()
  trunk:add(gater)
  trunk:add(experts)

  model = nn.Sequential()
  model:add(trunk)
  model:add(nn.MixtureTable())
  return model
end

function train(model,train_dataset,valid_dataset,max_epochs)
  
  optim_func=optim.rmsprop
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
    train_loss = train_loss / train_dataset:size()
    valid_loss = valid_loss / valid_dataset:size()

    train_confusion:updateValids()
    valid_confusion:updateValids()
    logger:add({train_loss,valid_loss})
    --[[
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
    --]]
    if best_validation_error>valid_loss then
      if i-last_save >=opt.save_gap then
        last_save=i
        os.execute('rm ./model/model.dat')
        torch.save('./model/model.dat',{['epoch']=i,
            ['train_accuracy']=train_confusion.totalValid,
            ['train_loss']=train_loss,
            ['valid_accuracy']=valid_confusion.totalValid,
            ['valid_loss']=valid_loss,
            ['model']=model})
      end
      best_validation_error=valid_loss
    end
  end
  return best_validation_error
end

function hyper()

  math.randomseed( os.time() )
--[[GPU or CPU]]--
  if opt.cuda then
    require 'cutorch'
    require 'cunn'
    --torch.setdefaulttensortype('torch.CudaTensor')
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

--[[GPU or CPU]]--
  if opt.cuda then
    criterion:cuda()
    if train_dataset then
      train_dataset:type('cuda')
    end
    if valid_dataset then
      valid_dataset:type('cuda')
    end
  end
  local best_valid_error=math.huge
  for i=1,1000 do
    local model = getModel()
    if opt.cuda then
      model:cuda()
    end
    if opt.sparse_init then
      print('Using sparse init')
      sparseReset(model)
    end
    valid_error=train(model,train_dataset,valid_dataset)
    print('cur valid error: '..valid_error)
    if valid_error<best_valid_error then
      best_valid_error=valid_error
      os.execute('rm ./model/model_best.dat')
      os.execute('mv ./model/model.dat ./model/model_best.dat')
      print('model saved!')
    end
    print('best: '..best_valid_error)
    collectgarbage()
  end

end

hyper()