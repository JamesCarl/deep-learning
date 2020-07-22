{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeApplications #-}
module Lib.Example.LinearRegression
  ( linearRegression
  )
where
import           Control.Monad                  ( forM_
                                                , when
                                                )
import           Control.Monad.IO.Class         ( liftIO )
import           Data.Int                       ( Int32
                                                , Int64
                                                )
import           Data.List                      ( genericLength )
import qualified Data.Text.IO                  as T
import qualified Data.Vector                   as V

import qualified TensorFlow.Core               as TF
import qualified TensorFlow.Ops                as TF
                                         hiding ( initializedVariable
                                                , zeroInitializedVariable
                                                )
import qualified TensorFlow.Variable           as TF
import qualified TensorFlow.Minimize           as TF

import           Control.Monad                  ( replicateM
                                                , replicateM_
                                                )
import           System.Random                  ( randomIO )
import           Test.HUnit                     ( assertBool )
import           Lib.Example.LinearRegression.Data
import qualified TensorFlow.Core               as TF
import qualified TensorFlow.GenOps.Core        as TF
                                         hiding ( placeholder )
import qualified TensorFlow.Minimize           as TF
import qualified TensorFlow.Ops                as TF
                                         hiding ( initializedVariable )
import qualified TensorFlow.Variable           as TF
import qualified Data.Vector                   as V


data Model = Model {
      train :: TF.TensorData Float
            -> TF.TensorData TimeType
            -> TF.Session ()
    , infer ::  TF.TensorData Float
            -> TF.Session (V.Vector TimeType)
    }

createModel :: TF.Build Model
createModel = do
  let batchSize = -1
  sizes <- TF.placeholder [batchSize]
  times <- TF.placeholder [batchSize]
  w     <- TF.initializedVariable 0
  b     <- TF.initializedVariable 0
  let yHat = ((sizes `TF.mul` TF.readValue w) `TF.add` TF.readValue b)
      loss = TF.square (yHat `TF.sub` times)
  predict <-
    TF.render @TF.Build @TimeType
      $ ((sizes `TF.mul` TF.readValue w) `TF.add` TF.readValue b)
  trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
  return Model
    { train = \sizesFeed timesFeed -> TF.runWithFeeds_
                [TF.feed sizes sizesFeed, TF.feed times timesFeed]
                trainStep
    , infer = \sizesFeed -> TF.runWithFeeds [TF.feed sizes sizesFeed] predict
    }


linearRegression :: IO ()
linearRegression = TF.runSession $ do
  model <- TF.build createModel
  let sizes     = TF.encodeTensorData [100] $ sizeMB trainData
  let times     = TF.encodeTensorData [100] $ timeSec trainData
  let testSizes = TF.encodeTensorData [100] $ sizeMB testData
  let testTimes = TF.encodeTensorData [100] $ timeSec testData
  -- Functions for generating batches.
  -- Train.
  forM_ ([0 .. 200] :: [Int]) $ \i -> train model sizes times
  testPreds <- infer model testSizes
  liftIO $ forM_ ([0 .. 3] :: [Int]) $ \i -> do
    putStrLn ""
    putStrLn $ "expected " ++ show (V.toList (timeSec testData) !! i)
    putStrLn $ "     got " ++ show (testPreds V.! i)
