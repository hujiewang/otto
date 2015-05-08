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
  if fname=='train.csv' then
    shuffle=torch.randperm(size)
    shuffle_inputs=torch.Tensor(size,93 )
    shuffle_targets=torch.Tensor(size)
    for i=1,size do
      shuffle_inputs[i]=inputs[shuffle[i]];
      if targets then
        shuffle_targets[i]=targets[shuffle[i]];
      end
    end

    local k=math.floor(size*0.08);
    valid_inputs=shuffle_inputs[{{1,k}}]
    valid_targets=shuffle_targets[{{1,k}}]
    train_inputs=shuffle_inputs[{{k+1,size}}]
    train_targets=shuffle_targets[{{k+1,size}}]
    return DataSet(train_inputs,train_targets,opt),DataSet(valid_inputs,valid_targets,opt)
  end
  return DataSet(inputs,nil,opt)
end