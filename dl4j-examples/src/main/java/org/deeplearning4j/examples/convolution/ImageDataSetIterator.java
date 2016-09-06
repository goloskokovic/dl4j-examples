/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.deeplearning4j.examples.convolution;

import java.io.IOException;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 *
 * @author gola
 */
public class ImageDataSetIterator extends BaseDatasetIterator {
           
        public ImageDataSetIterator(int batch, int numExamples, String folderPath, int width, int height, int depth) throws IOException {
            super(batch, numExamples, new ImageDataFetcher(folderPath, width, height, depth));
        }
        
        public ImageDataSetIterator(int batch, ImageDataFetcher dataFetcher) throws IOException {
            super(batch, dataFetcher.NUM_EXAMPLES, dataFetcher);
        }
    }
