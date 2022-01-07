#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

double angle(Point s, Point e, Point f) {

    double v1[2],v2[2];
    v1[0] = s.x - f.x;
    v1[1] = s.y - f.y;
    v2[0] = e.x - f.x;
    v2[1] = e.y - f.y;
    double ang1 = atan2(v1[1], v1[0]);
    double ang2 = atan2(v2[1], v2[0]);

    double ang = ang1 - ang2;
    if (ang > CV_PI) ang -= 2*CV_PI;
    if (ang < -CV_PI) ang += 2*CV_PI;

    return ang*180/CV_PI;


}

int main(int argc, char* argv[])
{
	Mat frame, roi, fgMask;
	VideoCapture cap;

  vector<vector<Point> > contours;
  
	namedWindow("Frame");
  namedWindow("ROI");
  namedWindow("Foreground Mask");

  Rect rect(400,100,200,200);

  Ptr<BackgroundSubtractor> pBackSub=createBackgroundSubtractorMOG2();

  int learning_rate = -1;

  //Si queremos grabar video debemos ejecutar pasando la opcion -r
  if(argc > 1 && strcmp(argv[1], "-r") == 0) {
    //abrimos la camara(se le pasa el valor del index de la camara en caso de haber mas de 1)
    cap.open(1);

    if (!cap.isOpened()) {
		  printf("Error opening cam\n");
		  return -1;
	  }

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    cout << "Grabando" << endl;
    int codec = VideoWriter::fourcc('M','J','P','G');
    VideoWriter video("out.avi",codec,20, Size(frame_width,frame_height));

    while(true) {

		  cap>>frame;
		  flip(frame,frame,1);

      //Copiamos la region dentro del rect a la ventana roi
      frame(rect).copyTo(roi);

      //Imprimimos un rectangulo rojo en la imagen para saber donde esta la region de interes
      rectangle(frame, rect,Scalar(255,0,0));

      //Sustraemos el fondo usando una mascara binaria
      pBackSub->apply(roi, fgMask, learning_rate);

      //Obtenemos los contornos de la imagen binaria sin el fondo
      findContours(fgMask,contours,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
      vector<int> hull;
      int max_size_contour_index = 0;
      //Buscamos el controno de tamaño mayor pues podemos presumir que este será el correspondiente a la mano
      for (size_t i = 0; i < contours.size(); i++) {
        if(contours[i].size() > contours[max_size_contour_index].size()) {
          max_size_contour_index = i;
        }
      }

      //Dibujamos el controno de la mano.
      drawContours(roi, contours, max_size_contour_index, Scalar(0,255,0),3);

      //Calculamos la malla convexa de la mano
      convexHull(contours[max_size_contour_index], hull, false, false);
      sort(hull.begin(),hull.end(),greater <int>());
        

      vector<Vec4i> defects;
      //Obtenemos los defectos de convexidad del contorno mas grande con respecto de la malla convexa
      convexityDefects(contours[max_size_contour_index], hull, defects);

      for (int i = 0; i < defects.size(); i++) {
        Point s = contours[max_size_contour_index][defects[i][0]];
        Point e = contours[max_size_contour_index][defects[i][1]];
        Point f = contours[max_size_contour_index][defects[i][2]];
        float depth = (float)defects[i][3] / 256.0;
        double ang = angle(s,e,f);
        //Para filtrar los defectos de convexidad buscamos aquellos que 
        //tengan una distancia mayor a 30 pixeles y un angulo menor de 95 grados.
        if(depth > 30 && ang < 95) {
          circle(roi, f,5,Scalar(0,0,255),-1);
          line(roi,s,e,Scalar(255,0,0),2);
        }
      }

      //Calculamos y dibujamos el boundinrect que contiene al contorno de la mano
      Rect boundRect = boundingRect(contours[max_size_contour_index]);
      rectangle(roi,boundRect,Scalar(0,0,255),3);

      //Mostramos las ventanas
		  imshow("Frame",frame);
      imshow("ROI",roi);
      imshow("Foreground Mask",fgMask);

      //Añadimos el frame al video
      video.write(frame);

		  int c = waitKey(40);
      //Si pulsamos s fijamos el learning rate del background substractor a 0 para que no añada nuevos elementos al fondo
      if ((char)c =='s') learning_rate = 0;
      //Si pulsamos q salimos del bucle de ejecucion liberando el video antes
		  if ((char)c =='q') {
        video.release();
        break;
      } 
	  }
    //Ejecucion procesando archivo de video
  } else if(argc > 2 && strcmp(argv[1], "-a") == 0) {
    cout << "Leyendo de archivo" << endl;
    string video_name = argv[2];
    //Abrimos el video cuyo nombre recibimmos como argumento de ejecucion
    cap.open(video_name);

    if (!cap.isOpened()) {
		  printf("Error opening video\n");
		  return -1;
	  }

    while(true) {
      
		  cap>>frame;
      //Cuando se acabe el video se para la ejecucion
      if (frame.empty()) break;

      //Copiamos la region dentro del rect a la ventana roi
      frame(rect).copyTo(roi);

      //Sustraemos el fondo usando una mascara binaria
      pBackSub->apply(roi, fgMask, learning_rate);

      //Obtenemos los contornos de la imagen binaria sin el fondo
      findContours(fgMask,contours,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
      vector<int> hull;
      int max_size_contour_index = 0;
      //Buscamos el controno de tamaño mayor pues podemos presumir que este será el correspondiente a la mano
      for (size_t i = 0; i < contours.size(); i++) {
        if(contours[i].size() > contours[max_size_contour_index].size()) {
          max_size_contour_index = i;
        }
      }

      //Dibujamos el controno de la mano.
      drawContours(roi, contours, max_size_contour_index, Scalar(0,255,0),3);

      //Calculamos la malla convexa de la mano
      convexHull(contours[max_size_contour_index], hull, false, false);
      sort(hull.begin(),hull.end(),greater <int>());
        

      vector<Vec4i> defects;
      //Obtenemos los defectos de convexidad del contorno mas grande con respecto de la malla convexa
      convexityDefects(contours[max_size_contour_index], hull, defects);

      for (int i = 0; i < defects.size(); i++) {
        Point s = contours[max_size_contour_index][defects[i][0]];
        Point e = contours[max_size_contour_index][defects[i][1]];
        Point f = contours[max_size_contour_index][defects[i][2]];
        float depth = (float)defects[i][3] / 256.0;
        double ang = angle(s,e,f);
        //Para filtrar los defectos de convexidad buscamos aquellos que 
        //tengan una distancia mayor a 30 pixeles y un angulo menor de 95 grados.
        if(depth > 30 && ang < 95) {
          circle(roi, f,5,Scalar(0,0,255),-1);
          line(roi,s,e,Scalar(255,0,0),2);
        }
      }

      //Calculamos y dibujamos el boundinrect que contiene al contorno de la mano
      Rect boundRect = boundingRect(contours[max_size_contour_index]);
      rectangle(roi,boundRect,Scalar(0,0,255),3);

      //Mostramos las ventanas
		  imshow("Frame",frame);
      imshow("ROI",roi);
      imshow("Foreground Mask",fgMask);

		  int c = waitKey(40);
      //Si pulsamos s fijamos el learning rate del background substractor a 0 para que no añada nuevos elementos al fondo
      if ((char)c =='s') learning_rate = 0;
      //Si pulsamos q salimos del bucle de ejecucion liberando el video antes
		  if ((char)c =='q') break;
	  }
    //Ejecucion sin leer ni grabar video
  }else {
    //abrimos la camara(se le pasa el valor del index de la camara en caso de haber mas de 1)
    cap.open(1);

    if (!cap.isOpened()) {
		  printf("Error opening cam\n");
		  return -1;
	  }

    while(true) {

		  cap>>frame;
      flip(frame,frame,1);

      //Copiamos la region dentro del rect a la ventana roi
      frame(rect).copyTo(roi);

      //Imprimimos un rectangulo rojo en la imagen para saber donde esta la region de interes
      rectangle(frame, rect,Scalar(255,0,0));

      Point p1(0, 0);
      Point p2(700, 50);

      rectangle(frame, p1, p2, Scalar(0,0,0), -1);

      //Sustraemos el fondo usando una mascara binaria
      pBackSub->apply(roi, fgMask, learning_rate);

      //Obtenemos los contornos de la imagen binaria sin el fondo
      findContours(fgMask,contours,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
      vector<int> hull;
      int max_size_contour_index = 0;
      //Buscamos el controno de tamaño mayor pues podemos presumir que este será el correspondiente a la mano
      for (size_t i = 0; i < contours.size(); i++) {
        if(contours[i].size() > contours[max_size_contour_index].size()) {
          max_size_contour_index = i;
        }
      }

      //Dibujamos el controno de la mano.
      drawContours(roi, contours, max_size_contour_index, Scalar(0,255,0),3);

      //Calculamos la malla convexa de la mano
      convexHull(contours[max_size_contour_index], hull, false, false);
      sort(hull.begin(),hull.end(),greater <int>());
        

      vector<Vec4i> defects;
      //Obtenemos los defectos de convexidad del contorno mas grande con respecto de la malla convexa
      convexityDefects(contours[max_size_contour_index], hull, defects);

      for (int i = 0; i < defects.size(); i++) {
        Point s = contours[max_size_contour_index][defects[i][0]];
        Point e = contours[max_size_contour_index][defects[i][1]];
        Point f = contours[max_size_contour_index][defects[i][2]];
        float depth = (float)defects[i][3] / 256.0;
        double ang = angle(s,e,f);
        //Para filtrar los defectos de convexidad buscamos aquellos que 
        //tengan una distancia mayor a 30 pixeles y un angulo menor de 95 grados.
        if(depth > 30 && ang < 95) {
          circle(roi, f,5,Scalar(0,0,255),-1);
          line(roi,s,e,Scalar(255,0,0),2);
        }
      }

      //Calculamos y dibujamos el boundinrect que contiene al contorno de la mano
      Rect boundRect = boundingRect(contours[max_size_contour_index]);
      rectangle(roi,boundRect,Scalar(0,0,255),3);

      // Point p1(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT) - 100);
      // Point p2(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));

      // rectangle(frame, p1, p2, Scalar(255,255,255), -1);

      //Mostramos las ventanas
		  imshow("Frame",frame);
      imshow("ROI",roi);
      imshow("Foreground Mask",fgMask);

		  int c = waitKey(40);
      //Si pulsamos s fijamos el learning rate del background substractor a 0 para que no añada nuevos elementos al fondo
      if ((char)c =='s') learning_rate = 0;
      //Si pulsamos q salimos del bucle de ejecucion liberando el video antes
		  if ((char)c =='q') break;
	  }
  }
  //Una vez acabada la ejecucion liberamos los recursos
	cap.release();
	destroyAllWindows();
}