require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

onserver = true
local sanitize = require('sanitize')


----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp[[
  -s,--save          (default "/media/data/daoyuan/16824/models")      subdirectory to save logs
  --saveFreq         (default 1)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -r,--learningRate  (default 0.001)      learning rate
  -b,--batchSize     (default 10)         batch size
  -m,--momentum      (default 0.9)         momentum term of adam
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  --scale            (default 512)          scale of images to train on
  --epochSize        (default 500)        number of samples per epoch
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --weightDecay      (default 0.0005)        weight decay
  --classnum         (default 40)    
  --classification   (default 1)
]]
if onserver then
    opt.save = '/home/daoyuanj/models'
end
if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.loadSize  = opt.scale 
-- TODO: setup the output size 
opt.labelSize = 16


opt.manualSeed = torch.random(1,10000) 
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}
opt.outDim =  opt.classnum


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


if opt.network == '' then
  ---------------------------------------------------------------------
  -- TODO: load pretrain model and add some layers, let's name it as model_FCN for the sake of simplicity
  -- hint: you might need to add large padding in conv1 (perhaps around 100ish? )
  -- hint2: use ReArrange instead of Reshape or View
  -- hint3: you might need to skip the dropout and softmax layer in the model
    local network = '/media/data/daoyuan/16824/3d/models/AlexNet'
    if onserver then
        network = '/scratch/16824/3d/models/AlexNet'
    end

    print('<trainer> reloading previously trained network: ' .. network)
    model_FCN = torch.load(network)
    model_FCN:get(1).padH = 100
    model_FCN:get(1).padW = 100

--    local newConv1 = nn.SpatialConvolution(3,96,11,11,4,4,100,100)
--    local oldConv1 = model_FCN:get(1)
--    local w, gw = oldConv1:parameters()
--    newConv1.weight = w
--    newConv1.gradWeight = gw
--    newConv1.bias = oldConv1.bias
--    newConv1.gradBias = oldConv1.gradBias -- why we can not change the padding directly???
--
--    model_FCN:remove(1)
--    model_FCN:insert(newConv1, 1)

    model_FCN:remove(24) -- remove dropout
    model_FCN:remove(23) -- remove FC layer
    model_FCN:remove(22) -- remove softmax-1000
    model_FCN:remove(19) -- remove another dropout
    model_FCN:remove(16) -- remove view
    model_FCN:add(nn.SpatialConvolution(4096, opt.classnum, 1, 1, 1, 1))
--    model_FCN:add(nn.ReArrange()) -- append ReArrange
    model_FCN:add(nn.Transpose({2,3},{3,4}))
    model_FCN:add(nn.View(-1, opt.classnum))
--    model_FCN:add(nn.LogSoftMax()) -- append log softmax

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_FCN = tmp.FCN
end

-- print networks
print('fcn network:')
print(model_FCN)


-- TODO: loss function
-- TODO: retrieve parameters and gradients
-- TODO: setup dataset, use data.lua
-- TODO: setup training functions, use fcn_train_cls.lua
-- TODO: convert model and loss function to cuda

criterion = nn.CrossEntropyCriterion()

if opt.gpu then
    model_FCN:cuda()
    criterion:cuda()
    print('Copy model 2 gpu')
end

parameters_FCN,gradParameters_FCN = model_FCN:getParameters()

paths.dofile('data.lua')
fcn = paths.dofile('fcn_train_cls.lua')




local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local function train()
   print('\n<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')
  
   model_FCN:training()
   batchNumber = 0
   for i=1,opt.epochSize do
      donkeys:addjob(
         function()
            return makeData_cls(trainLoader:sample(opt.batchSize))
         end,
         fcn.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()

end

epoch = 1
-- training loop
while true do
  -- train/test
  train()

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('fcn_%d.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, { FCN = sanitize(model_FCN), opt = opt})
  end

  epoch = epoch + 1

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
