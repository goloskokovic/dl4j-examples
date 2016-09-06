/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.deeplearning4j.examples.convolution;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author gola
 */
public class ImageDataFetcher extends BaseDataFetcher {

        Map<Integer, String> classLabels = new HashMap<>();
        List<String> imagePaths;
        
        public int NUM_EXAMPLES;
        int depth;              
        
        public ImageDataFetcher(String folderPath, int width, int height, int depth) throws IOException {
        
            loadImagePaths(folderPath);
            
            // these are always needed to be set
            totalExamples = NUM_EXAMPLES = imagePaths.size();
            numOutcomes = classLabels.size();
            cursor = 0;
            inputColumns = width * height * depth;
            this.depth = depth;
        }
        
        @Override
        public void fetch(int numExamples) {
            
            //we need to ensure that we don't overshoot the number of examples total
            List<org.nd4j.linalg.dataset.DataSet> toConvert = new ArrayList<>(numExamples);
            for( int i=0; i<numExamples; i++, cursor++ ){
                if(!hasMore()) {
                    break;
                }
                
                String[] image_class = imagePaths.get(cursor).split(" ");
                String imagePath = image_class[0];
                String iclass = image_class[1];
                int output = Integer.parseInt(iclass);

                BufferedImage image = null;
                INDArray in = null;                
                
                try {
                    
                    image = ImageIO.read(new File(imagePath));
                    
                } catch (IOException ex) { 
                    ex.printStackTrace();
                }
                
                int[] img = this.depth > 1 ? getRgbByteArray(image) : getByteArray(image);
                
                in = Nd4j.create(1, img.length);
                for( int j=0; j<img.length; j++ ) 
                    in.putScalar(j, ((int)img[j]) & 0xFF);  //byte is loaded as signed -> convert to unsigned
                
                in.divi(255);
                
                INDArray out = createOutputVector(output);

                toConvert.add(new org.nd4j.linalg.dataset.DataSet(in,out));
            }
            initializeCurrFromList(toConvert);
            
        }
        
        @Override
        public void reset() {
            cursor = 0;
            curr = null;
            //if(shuffle) MathUtils.shuffleArray(order, rng);
        }

        @Override
        public DataSet next() {
            DataSet next = super.next();
            return next;
        }
        
        void loadImagePaths(String folderPath) throws IOException {
    
            if (folderPath.endsWith("train/"))
                imagePaths = Files.readAllLines(Paths.get(folderPath + "train.txt"));
            else if (folderPath.endsWith("test/"))
                imagePaths = Files.readAllLines(Paths.get(folderPath + "test.txt"));
            
            
            List<String> labels = Files.readAllLines(Paths.get(folderPath + "labels.txt"));
        
            for(int i=0; i<labels.size(); i++)
                classLabels.put(i, labels.get(i));
        }
        
        int[] getRgbByteArray(BufferedImage image) {

            int w = image.getWidth();
            int h = image.getHeight();
            int BLOCK_SIZE = 3;

            int[] pixels = new int[w * h];
            image.getRGB(0, 0, w, h, pixels, 0, w);

            int[] rgbBytes = new int[w * h * BLOCK_SIZE];

            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    int index = r * w + c;
                    int indexRgb = r * w * BLOCK_SIZE + c * BLOCK_SIZE;

                    rgbBytes[indexRgb] = (byte)((pixels[index] >> 16) &0xff);
                    rgbBytes[indexRgb + 1] = (byte)((pixels[index] >> 8) &0xff);
                    rgbBytes[indexRgb + 2] = (byte)(pixels[index] &0xff);
                }
            }

            return rgbBytes;
        }
        
        int[] getByteArray(BufferedImage img) {
            
            int w = img.getWidth(null);
            int h = img.getHeight(null);
            int[] rgbs = new int[w * h];
            img.getRGB(0, 0, w, h, rgbs, 0, w);
            
            return rgbs;
        }     
        
    }
