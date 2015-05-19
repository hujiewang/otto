models={}

model = nn.Sequential()

model:add(nn.BatchNormalization(50))
model:add(nn.Dropout(0.12))

model:add(nn.Linear(50,512))
model:add(nn.ReLU())
model:add(nn.BatchNormalization(512))
model:add(nn.Dropout())

model:add(nn.Linear(512,512))
model:add(nn.ReLU())
model:add(nn.BatchNormalization(512))
model:add(nn.Dropout())

model:add(nn.Linear(512,512))
model:add(nn.ReLU())
model:add(nn.BatchNormalization(512))
model:add(nn.Dropout())

model:add(nn.Linear(512,9))

model:add(nn.LogSoftMax())