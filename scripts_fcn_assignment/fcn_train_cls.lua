require 'torch'
require 'optim'
require 'pl'
require 'paths'

local fcn = {}

local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- put the labels for each batch in targets
local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- training function
function fcn.train(inputs_all)
  cutorch.synchronize()
  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize 
    local loss = -1
  -- TODO: implemnet the training function
    local evalFCN = function (x)
        collectgarbage() -- why?
        -- what is parameter_FCN
        if x~= parameters_FCN then
            parameters_FCN:copy(x)
        end

        gradParameters_FCN:zero() -- reset gradients

        -- foward pass
        local outputs = model_FCN:forward(inputs)
        loss = criterion:forward(outputs, targets)
--        print (string.format('FCN loss: %f', f))

        local df_samples = criterion:backward(outputs, targets)
        model_FCN:backward(inputs, df_samples)

        return f, gradParameters_FCN
    end

    inputs:copy(inputs_all[1])
    targets:copy(inputs_all[2])
--    print(#inputs)
--    print(#inputs_all[2])
--    print(#targets)
    optim.sgd(evalFCN, parameters_FCN, optimState)

  batchNumber = batchNumber + 1
  cutorch.synchronize(); collectgarbage();
  print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f, loss %.7f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime, loss))
  dataTimer:reset()

end



return fcn


