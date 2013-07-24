package mytest;

import com.googlecode.javacv.cpp.opencv_contrib;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_highgui;
import com.googlecode.javacv.cpp.opencv_imgproc;
import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Algorithm;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SexDetector {

    private final static String imgCatalog = "d:\\my\\img\\list.csv";
    private final static String testImgPath = "d:\\my\\img\\aniston.jpg";

    public static void main( String[] args ) throws FileNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new SexDetector().run();
    }


    public void run() throws FileNotFoundException {
        System.out.println("\nRunning Sex Detection...");

        List<opencv_core.IplImage> images = new ArrayList<opencv_core.IplImage>(50);

        List<Integer> labels = new ArrayList<Integer>(50);
        readCsv(images, labels);

        // The following lines create an Fisherfaces model for
        // face recognition and train it with the images and
        // labels read from the given CSV file.
        opencv_contrib.FaceRecognizer model = opencv_contrib.createFisherFaceRecognizer();

        opencv_core.MatVector imagesVector = new opencv_core.MatVector(images.size());
        for(int i = 0; i < images.size(); i++) imagesVector.put(i, normilizeImg(images.get(i)));

        int[] labelsArr = new int[labels.size()];
        for(int i = 0; i < labels.size(); i++) labelsArr[i] = labels.get(i);

        long startAt = System.currentTimeMillis();
        model.train(imagesVector, labelsArr);
        System.out.println("Time spent for training is " + (System.currentTimeMillis() - startAt));

        opencv_core.IplImage testImg = normilizeImg( opencv_highgui.cvLoadImage(new File(testImgPath).getAbsolutePath()) );
        int prediction = model.predict( testImg );

        System.out.println( String.format("Person on image is %s", prediction == 0 ? "man" : "woman") );
    }

    private opencv_core.IplImage normilizeImg(opencv_core.IplImage originalImage) {
        opencv_core.CvSize size = opencv_core.cvSize(1024, 1446);

        opencv_core.IplImage tmpImg = opencv_core.IplImage.create(size, originalImage.depth(), originalImage.nChannels());
        opencv_imgproc.cvResize(originalImage, tmpImg, opencv_imgproc.CV_INTER_LINEAR);
        opencv_core.IplImage tmpImg2 = opencv_core.IplImage.create(size, 8, 1);
        opencv_imgproc.cvCvtColor(tmpImg, tmpImg2, opencv_imgproc.CV_RGB2GRAY);

        return tmpImg2;
    }

    private void readCsv(List<opencv_core.IplImage> images, List<Integer> labels) throws FileNotFoundException {
        BufferedReader br = new BufferedReader(new FileReader(imgCatalog));
        try {
            String line = br.readLine();
            while (line != null) {
                String[] s = line.split(";");
                images.add( opencv_highgui.cvLoadImage(new File(s[0]).getAbsolutePath()) );
                labels.add(Integer.valueOf(s[1]));
                line = br.readLine();
            }

        } catch(IOException e) {
            e.printStackTrace();
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}