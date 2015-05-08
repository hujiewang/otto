require 'cutorch'
require 'cunn'
require 'optim'
require 'xlua'
require 'DataSet'
require 'image'
require 'load_data'
require 'csvigo'
require 'tools'

pre_opt={
  batch_size = 8192,
}
function predict(model_fname,dataset,cuda)

  local model_data=torch.load(model_fname)
  local model=model_data['model']
  local loss = model_data['valid_loss']
  if loss then
    print('model valid loss: '..loss)
  end
  -- For preserving order of the table
  rv_idx={}
  table.insert(rv_idx,'id')
  for i=1,9 do
    table.insert(rv_idx,'Class_'..i)
  end

  local rv={}
  for k,v in pairs(rv_idx) do
    rv[v]={}
  end
  
  for batch = 1,math.ceil(dataset:size()/pre_opt.batch_size)do    
  xlua.progress(batch,math.ceil(dataset:size()/pre_opt.batch_size))
  local input,_=dataset:getBatch(batch)
  if cuda then
    input=input:cuda()
  end
  local output = model:forward(input)
  output:exp()

  local s=(batch-1)*pre_opt.batch_size+1
  local e=math.min(batch*pre_opt.batch_size,dataset:size())

  for k=s,e do
    table.insert(rv['id'],k)
    for i=1,9 do
      table.insert(rv['Class_'..i],output[k-s+1][i])
    end
  end
end
return rv
end

function predictMulti(models,va)
  print('')
  print('============================================================')
  print('Predicting...')
  print('')
  local dataset
  if va then
    data=torch.load('./data/data.dat')
    dataset=data['valid']
  else
    dataset=load_data('test.csv')
  end
  dataset.opt=pre_opt
  
  local stand=torch.load('./data/standardize.dat')
  local mean=stand['mean']
  local std=stand['std']
  standardize(nil,nil,dataset,mean,std)
  
  local data=nil
  for i=1,#models do
    local rv=predict(models[i][1],dataset,models[i][2])
    if not data then
      data=rv
    else
      for i=1,9 do
        for j=1,#data['Class_'..i] do
          data['Class_'..i][j]=data['Class_'..i][j]+rv['Class_'..i][j]
        end
      end
    end
  end
  for i=1,9 do
    for j=1,#data['Class_'..i] do
      data['Class_'..i][j]=data['Class_'..i][j]/#models 
    end
  end
  if va then
    local loss=0
    for i=1,#data['Class_'..1] do
      loss=loss+math.log(data['Class_'..dataset.target[i]][i])
    end
    loss =loss/#data['Class_'..1]
    loss = -loss
    print('loss: '..loss)
  end
  csvigo.save({path='./results/results.csv',data=data})
  print('Saved!')
end

function predictMulti2(outputs)
  local data=nil
  for i=1,#outputs do
    local rv=csvigo.load(outputs[i])
    if not data then
      data=rv
    else
      for i=1,9 do
        for j=1,#data['Class_'..i] do
          data['Class_'..i][j]=data['Class_'..i][j]+rv['Class_'..i][j]
        end
      end
    end
  end
  for i=1,9 do
    for j=1,#data['Class_'..i] do
      data['Class_'..i][j]=data['Class_'..i][j]/#outputs
    end
  end
  csvigo.save({path='./results/results.csv',data=data})
  print('Saved!')
end
--[[
predictMulti({
  {'./model/model_6.dat',true},
  {'./model/model_8.dat',false},
  --{'./model/model_11.dat',true},
  --{'./model/model_13.dat',true},
  --{'./model/model_14.dat',true},
  --{'./model/model_15.dat',true},
  },true)
----predictMulti2({'./results_6.csv','./results_7.csv','./results_8.csv','./results_11.csv','./results_13.csv','./results_14.csv','./results_15.csv'})
--]]

--[[
models={}
models_csv={}
for i=1,6 do
  table.insert(models,'./model/'..i..'.dat')
  table.insert(models_csv,'./RF_'..i..'.csv')
end


predictMulti(models,true,true)
--]]