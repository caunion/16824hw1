--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'

paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "../cache"
os.execute('mkdir -p '..cache)
local trainCache = paths.concat(cache, 'trainCache_assignment2.t7')


-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '../logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}
local labelSampleSize = {3, opt.labelSize}

-- read the codebook (40 * 3)

local codebooktxt = '/media/data/daoyuan/16824/3d/list/codebook_40.txt'
if onserver then
    codebooktxt = '/scratch/16824/3d/list/codebook_40.txt'
end
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 

  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()
end


local div_num, sub_num
div_num = 127.5
sub_num = -1


local function loadImage(filepath)
    if not paths.filep(filepath) then
        print('opps, image not found!!!' .. filepath )
    end
   local input = image.load(filepath, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize)
   input = input * 255
   return input
end


local function loadLabel_high(filepath)
    if not paths.filep(filepath) then
        print('opps, normal label not found!!!' .. filepath )
    end
   local input = image.load(filepath, 3, 'float')
   input = image.scale(input, opt.labelSize, opt.labelSize )
   input = input * 255
   return input
end



function makeData_cls(img, label)
  -- TODO: the input label is a 3-channel real value image, quantize each pixel into classes (1 ~ 40)
  -- resize the label map from a matrix into a long vector
  -- hint: the label should be a vector with dimension of: opt.batchSize * opt.labelSize * opt.labelSize
    local shape = #label
    shape[2] = 40 -- since there 40 codes
--    local ret = torch.Tensor(40, shape[1], shape[3], shape[4])
--    print(label[1])
--    for i = 1, shape[2] do
--        local codemap = codebook[i]:repeatTensor(shape[1], shape[4], shape[3], 1):transpose(2, 4) -- c * H * W
--        print(string.format('codebook[i]: %f %f %f', codebook[i][1], codebook[i][2], codebook[i][3]))
--        print(string.format('codemap val: %f %f %f',codemap[1][1][1][1],codemap[1][2][1][1],codemap[1][3][1][1]))
--        print('label val')
--        print(string.format('label val: %f %f %f',label[1][1][1][1],label[1][2][1][1],label[1][3][1][1]))
--        print('temp val')
--        local temp = codemap:mul(label, 1.0)
--
--        print(temp[1][1][1][1])
--        print(temp[1][2][1][1])
--        print(temp[1][3][1][1])
--        print(temp:sum(2)[1][1][1][1])
--        ret[i] = temp:sum(2)
--    end
--
--
--    local temp = 0
--    temp, label =  ret:max(1)
--    print('ret[1][1] is:')
--    print(ret[1][1])
--    print('ret[2][1] is:')
--    print(ret[2][1])
--    print('temp is')
--    print(temp[1][1])
--    print('label is')
--    print(label[1][1])

--    print(label)
--    label = label:view(opt.batchSize * opt.labelSize * opt.labelSize)
      local temp = torch.Tensor(40,opt.batchSize * opt.labelSize * opt.labelSize)
       label = label:transpose(1,2):reshape(3, opt.batchSize * opt.labelSize * opt.labelSize)
--        print(#label)
--        print(#codebook)
      torch.mm(temp, codebook, label)
      temp, label = temp:max(1)
--    print(#label)
      label = label:reshape(opt.batchSize * opt.labelSize * opt.labelSize)
--      print(#label)
--      print(label)
----    print(#label)
----    print('label size')
    return {img, label}
end


function makeData_cls_pre(img, label)
  -- TODO: almost same as makeData_cls, need to convert img from RGB to BGR for caffe pre-trained model
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(2, perm)
    return makeData_cls(img, label)
end





--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, imgpath, lblpath)
   collectgarbage()
   local img = loadImage(imgpath)
   local label = loadLabel_high(lblpath)
   img:add( - 127.5 )
   label:div(div_num)
   label:add(sub_num)

   return img, label

end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
   trainLoader.labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      labelSampleSize = {3, labelSampleSize[2], labelSampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()



