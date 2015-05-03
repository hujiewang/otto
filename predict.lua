
function predict()
  print('')
  print('============================================================')
  print('Predicting...')
  print('')

  local model_data=torch.load('model.dat')
  local model=model_data['model']
  local dataset=load_data('test.csv')
  
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
  
  for i = 1,dataset:size() do    
    local output = model:forward(dataset.input[i])
    local y,idx=torch.max(output,1)
    table.insert(rv['id'],i)
    for i=1,9 do
      if idx[1] == i then
        table.insert(rv['Class_'..i],1)
      else
        table.insert(rv['Class_'..i],0)
      end
    end
  end
  csvigo.save({path='results.csv',data=rv,mode=query})
  print('Saved!')
end

--predict()