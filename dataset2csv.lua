require 'cutorch'
require 'cunn'
require 'optim'
require 'xlua'
require 'DataSet'
require 'image'
require 'load_data'
require 'csvigo'
require 'tools'

opt={
  createData = false,
  epochs = 100000,
  batch_size = 10000,
  predict = false,
  save_gap = 10,
  cuda=true,
  plot=false,
  sparse_init = true,
  standardize = true,
  --model_file = 'model.dat'
}

function toTable(dataset)
  rv={}
  rv['id']={}
  rv['target']={}
  for i=1,93 do
    rv['feat_'..i]={}
  end

  for i=1,dataset:size() do
    table.insert(rv['id'],i)
    table.insert(rv['target'],'Class_'..dataset.target[i])
    for j=1,93 do
      table.insert(rv['feat_'..j],dataset.input[i][j])
    end
  end
  return rv
end
function dataset2csv()
  data=torch.load('./data/data.dat')
  train_dataset=data['train']
  valid_dataset=data['valid']
  train_dataset.opt=opt
  valid_dataset.opt=opt
  print('Data has been loaded!')
  train_csv=toTable(train_dataset)
  valid_csv=toTable(valid_dataset)
  csvigo.save({path='train_data.csv',data=train_csv})
  csvigo.save({path='valid_data.csv',data=valid_csv})
  print('Saved!')
  t_csv=csvigo.load('train_data.csv')
  v_csv=csvigo.load('valid_data.csv')
  for i=1,1000 do
    for j=1,93 do
      assert(''..t_csv['feat_'..j][i]==''..train_dataset.input[i][j])
      assert(''..v_csv['feat_'..j][i]==''..valid_dataset.input[i][j])
      assert(t_csv['target'][i]==('Class_'..train_dataset.target[i]))
      assert(v_csv['target'][i]==('Class_'..valid_dataset.target[i]))
    end
  end
  print('checking done!')
end

dataset2csv()


