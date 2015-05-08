--[[
DataSet Class
--]]

do
  local DataSet = torch.class('DataSet')

--[[
input: Tensor
target: Tensor
--]]
  function DataSet:__init(input,target,opt)
    if target then
      assert(input:size(1)==target:size(1))
    end
    self.input=input
    self.target=target
    if self.input:dim() == 1 then
      self.input = self.input:reshape(1,self.input:size(1))
    end

    self.shuffle = randperm((#self.input)[1])
    self.opt=opt

  end
  function DataSet:type(new_type)
    if new_type == 'cuda' then
      self.input = self.input:cuda()
      if self.target then
        self.target = self.target:double():cuda()
      end
    elseif new_type == 'float' then
      self.input = self.input:float()
    elseif new_type == 'double' then
      self.input = self.input:double()
    end
  end
  function DataSet:getBatch(batch)
    local s=(batch-1)*self.opt.batch_size+1
    local e=math.min(batch*self.opt.batch_size,self:size())
    local inputs=self.input[{{s,e}}]
    local targets=nil
    if self.target then
      targets=self.target[{{s,e}}]
    end
    --for i=s,e do
      --print(self.shuffle[i],i)
      --inputs[i-s+1]=self.input[self.shuffle[i]]
      --if self.target then
        --targets[i-s+1]=self.target[self.shuffle[i]]
      --end
    --end
    
    return inputs,targets
  end

  function DataSet:shuffleData()
    self.shuffle = randperm((#self.input)[1])
  end
  function DataSet:shuffleComplete()
    self.shuffle = randperm(self:size())
    
    shuffle_inputs=self.input:clone()
    if self.target then
      shuffle_targets=self.target:clone()
    end
    for i=1,self:size() do
      shuffle_inputs[i]=self.input[self.shuffle[i]];
      if targets then
        shuffle_targets[i]=self.target[self.shuffle[i]];
      end
    end
    self.input=shuffle_inputs
    self.target=shuffle_targets
    collectgarbage()
  end
  function DataSet:size()
    return (#self.input)[1]
  end
end