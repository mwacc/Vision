package mytest;

import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_highgui;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App 
{
    private final static String imgCatalog = "d:\\my\\img\\list.csv";

    public static void main( String[] args ) throws FileNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new App().run();
    }


    public void run() throws FileNotFoundException {
        System.out.println("\nRunning DetectFaceDemo");

        // Create a face detector from the cascade file in the resources
        // directory.
        CascadeClassifier faceDetector = new CascadeClassifier(new File("C:\\opencv\\data\\lbpcascades\\lbpcascade_frontalface.xml").getPath());
        List<Mat> images = new ArrayList<Mat>(50);
        readCsv(images);

        int i = 0;
        for(Mat image : images) {
            // Detect faces in the image.
            // MatOfRect is a special container class for Rect.
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(image, faceDetections);

            System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

            // Draw a bounding box around each face.
            for (Rect rect : faceDetections.toArray()) {
                Core.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
            }

            // Save the visualized detection.
            String filename = "D:\\my\\img\\dest\\" + i + ".png";
            System.out.println(String.format("Writing %s", filename));
            Highgui.imwrite(filename, image);
            i++;
        }
    }

    private void readCsv(List<Mat> images) throws FileNotFoundException {
        BufferedReader br = new BufferedReader(new FileReader(imgCatalog));
        try {
            String line = br.readLine();
            while (line != null) {
                String[] s = line.split(";");
                images.add( Highgui.imread(new File(s[0]).getAbsolutePath()) );
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
