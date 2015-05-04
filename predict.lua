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
function predict()
  print('')
  print('============================================================')
  print('Predicting...')
  print('')

  local model_data=torch.load('./model/model.dat')
  local model=model_data['model']
  local stand=torch.load('standardize.dat')
  local mean=stand['mean']
  local std=stand['std']
  local loss = model_data['valid_loss']
  local dataset=load_data('test.csv')
  dataset.opt=pre_opt
  standardize(nil,nil,dataset,mean,std)
  print('model valid loss: '..loss)
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
    input=input:cuda()
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
  csvigo.save({path='results.csv',data=rv,mode=query})
  print('Saved!')
end

--predict()