{-#LANGUAGE ScopedTypeVariables #-}
module Main where

import CV.Image
import CV.ImageOp
import CV.ImageMathOp
import CV.ImageMath
import CV.Video
import CV.HighGUI
import Data.Maybe
import Utils.Stream

import Control.Monad
import Control.Monad.State
import Control.Monad.ST

import Data.List (init)
import Data.STRef

type MvAvgState = [Image RGB D32]

mvAvg :: Image RGB D32 -> State MvAvgState (Image RGB D32, Image RGB D32)
mvAvg img = do
  avg <- get
  case length avg of
    6 -> do
      put $ img:init avg
      return (head avg, img)
    _ -> do
      put $ img:avg
      return (head avg, img)
{-
proc img = do
  n <- evalState (mvAvg img) []
  return $ snd n -}

--mapSS (Monad m) => Stream m a -> (a -> s -> b) -> Stream m b
--mapSS stream first proc = foldS first (\a b -> let (s,v) = proc a b in v

uhacko stream mvAvgST = sideEffect (\ i -> showImage "a" i >> waitKey 1 >> return ())
      $ mapMS (\img -> stToIO $ do
        let gray = rgbToGray img
        lst <- readSTRef mvAvgST
        if length lst < 60
          then modifySTRef mvAvgST (gray:)
          else writeSTRef mvAvgST (gray:init lst)
        lst' <- readSTRef mvAvgST
        return $ grayToRGB $ averageImages lst'
      ) s1 --(empty $ getFrameSize cap)
    where s1 = sideEffect (\ i -> showImage "b" i >> waitKey 1 >> return ())
               $ stream


main = do
  makeWindow "a"
  makeWindow "b"
  Just cap <- captureFromCam 0
{-  getFrame cap >>= (showImage "a").fromJust
  waitKey 1-}
  mvAvgST <- stToIO $ newSTRef ([] :: [Image GrayScale D32])
  let stream = streamFromVideo cap
  o <- runLast (empty $ getFrameSize cap) $ uhacko stream mvAvgST
  saveImage "o.png" o
  destroyWindow "a"
  destroyWindow "b"

--  frms <- fmap (fmap fromJust) $ sequence $ replicate 30 $ getFrame cap
--  saveImage "out.png" $ foldr1 (#+) frms
