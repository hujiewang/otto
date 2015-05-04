--require 'nn'

--[[ Model #2
epoch = 227 of 100000 current loss = 1.9432304411337e-05D10h | Step: 2s124ms    
Training accuracy:	
87.19471547179	
Validation accuracy:	
82.092929292929	
--]]

model = nn.Sequential()

model:add(nn.Dropout(0.3))

model:add(nn.Linear(93,512))
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