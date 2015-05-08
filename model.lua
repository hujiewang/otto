local n = 3

experts = nn.ConcatTable()
for i = 1, n do
  local expert = nn.Sequential()
  expert:add(nn.BatchNormalization(93))
  expert:add(nn.Dropout(0.25))

  expert:add(nn.Linear(93,512))
  expert:add(nn.ReLU())
  expert:add(nn.BatchNormalization(512))
  expert:add(nn.Dropout())

  expert:add(nn.Linear(512,512))
  expert:add(nn.ReLU())
  expert:add(nn.BatchNormalization(512))
  expert:add(nn.Dropout())

  expert:add(nn.Linear(512,512))
  expert:add(nn.ReLU())
  expert:add(nn.BatchNormalization(512))
  expert:add(nn.Dropout())

  expert:add(nn.Linear(512,9))

  expert:add(nn.LogSoftMax())
  experts:add(expert)
end

gater = nn.Sequential()

gater:add(nn.Linear(93,512))
gater:add(nn.ReLU())

gater:add(nn.Linear(512,512))
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