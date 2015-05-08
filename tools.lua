function randperm(n)
  t={}
  for i=1,n do
    table.insert(t,i)
  end
  while n > 1 do
    local k = math.random(n)
    t[n], t[k] = t[k], t[n]
    n = n - 1
  end
  return t
end

function sparseReset(mod)
  if not mod.modules then
    if mod.weight and mod.weight:dim() == 2 then
        _sparseReset(mod.weight)
    end
    return
  end
  for k,v in pairs(mod.modules) do
    sparseReset(v)
  end
end

function _sparseReset(W, stdev)
   assert(W:dim() == 2, 
      "Model.sparseInit requires a tensor with two dims at arg 1")
   stdev = stdev or 1
   W:zero()
   local output_size, input_size = W:size(1), W:size(2)
   local sparse_init = math.min(math.ceil(input_size/2), 15)
   -- for each output unit:
   for i = 1, output_size do
      -- initialize self.sparse_init input weights:
      for j = 1, sparse_init do
         local idx = math.ceil(math.random() * input_size)
         while W[{i, idx}] ~= 0 do
            idx = math.ceil(math.random() * input_size)
         end
         W[{i, idx}] = torch.normal(0, stdev)
      end
   end
end

--[[ Normalizes features --]]
function standardize(train_dataset,valid_dataset,test_dataset,_mean,_std)
  mean={}
  std={}
  new_mean={}
  new_std={}
  if test_dataset then
    for i=1,test_dataset.input:size(2) do
      test_dataset.input:select(2,i):add(-_mean[i])
      test_dataset.input:select(2,i):div(_std[i])
    end
    return
  end
  for i=1,train_dataset.input:size(2) do
    mean[i] = train_dataset.input:select(2,i):mean()
    std[i] = train_dataset.input:select(2,i):std()
    train_dataset.input:select(2,i):add(-mean[i])
    train_dataset.input:select(2,i):div(std[i])
    new_mean[i]=train_dataset.input:select(2,i):mean()
    new_std[i]=train_dataset.input:select(2,i):std()
    if valid_dataset then
      valid_dataset.input:select(2,i):add(-mean[i])
      valid_dataset.input:select(2,i):div(std[i])
    end
  end
  --print('new mean')
  --print(new_mean)
  --print('new std')
  --print(new_std)
  os.execute('rm standardize.dat')
  torch.save('./data/standardize.dat',{['mean']=mean,['std']=std})
  print('Standardization data saved!')
end