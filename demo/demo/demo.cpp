#include <iostream>
#include <signal.h>
#include <stdlib.h> /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Yolo3Detection.h"
#include "send.h"
#include "ekf.h"
#include "trackutils.h"
#include "plot.h"
#include "tracker.h"

#define MAX_DETECT_SIZE 100

bool gRun;

void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[])
{

    std::cout << "detection\n";
    signal(SIGINT, sig_handler);

    char *net = "yolo3_coco4.rt";
    if (argc > 1)
        net = argv[1];
    char *input = "../demo/demo/data/single_ped_2.mp4";
    if (argc > 2)
        input = argv[2];
    char *pmatrix = "../demo/demo/data/proj_matrix_map_b.txt";
    if (argc > 3)
        pmatrix = argv[3];
    char *tiffile = "../demo/demo/data/map_b.tif";
    if (argc > 4)
        tiffile = argv[4];
    /*CAMID*/
    int CAM_IDX = 0;
    if (argc > 5)
        CAM_IDX = atoi(argv[5]);
    bool to_show = true;
    if (argc > 6)
        to_show = atoi(argv[6]);

    tk::dnn::Yolo3Detection yolo;
    yolo.init(net);
    yolo.thresh = 0.25;

    gRun = true;

    cv::VideoCapture cap(input);
    if (!cap.isOpened())
        gRun = false;
    else
        std::cout << "camera started\n";

    cv::Mat frame;
    cv::Mat dnn_input;
    cv::namedWindow("detection", cv::WINDOW_NORMAL);

    /*projection matrix*/

    int proj_matrix_read = 0;
    cv::Mat H(cv::Size(3, 3), CV_64FC1);

    /*GPS information*/
    double *adfGeoTransform = (double *)malloc(6 * sizeof(double));
    readTiff(tiffile, adfGeoTransform);

    /*socket*/
    int sock;
    int socket_opened = 0;

    /*Conversion for tracker, from gps to meters and viceversa*/
    geodetic_converter::GeodeticConverter gc;
    gc.initialiseReference(44.655540, 10.934315, 0);
    double east, north, up;
    double lat, lon, alt;

    /*tracker infos*/
    std::vector<Tracker> trackers;
    std::vector<Data> cur_frame;
    int initial_age = -5;
    int age_threshold = -20;
    int n_states = 5;
    float dt = 0.03;

    struct obj_coords *coords = (struct obj_coords *)malloc(MAX_DETECT_SIZE * sizeof(struct obj_coords));

    int frame_nbr = 0;
    while (gRun)
    {

        cap >> frame;
        if (!frame.data)
        {
            usleep(1000000);
            cap.open(input);
            printf("cap reinitialize\n");
            continue;
        }

        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        yolo.update(dnn_input);

        int coord_i = 0;

        int num_detected = yolo.detected.size();
        if (num_detected > MAX_DETECT_SIZE)
            num_detected = MAX_DETECT_SIZE;

        if (proj_matrix_read == 0)
        {
            read_projection_matrix(H, proj_matrix_read, pmatrix);
        }

        /*printf("%f %f %f \n%f %f %f\n %f %f %f\n\n", proj_matrix[0],proj_matrix[1],
            proj_matrix[2],proj_matrix[3],proj_matrix[4],proj_matrix[5],
            proj_matrix[6],proj_matrix[7],proj_matrix[8]);*/

        // draw dets
        for (int i = 0; i < num_detected; i++)
        {

            tk::dnn::box b = yolo.detected[i];
            int x0 = b.x;
            int x1 = b.x + b.w;
            int y0 = b.y;
            int y1 = b.y + b.h;
            int obj_class = b.cl;

            if (obj_class == 0 /*person*/ || obj_class == 1 /*bicycle*/ || obj_class == 2 /*car*/
                || obj_class == 3 /*motorbike*/ || obj_class == 5 /*bus*/)
            {
                convert_coords(coords, coord_i, x0 + b.w / 2, y1, obj_class, H, adfGeoTransform, frame_nbr);
                coord_i++;
            }

            float prob = b.prob;

            //std::cout<<obj_class<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
            cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[obj_class], 2);
        }

        cur_frame.clear();
        for (int i = 0; i < coord_i; i++)
        {
            //std::cout << "lat orig: " << coords[i].LAT << " lon orig: " << coords[i].LONG << std::endl;
            gc.geodetic2Enu(coords[i].LAT, coords[i].LONG, 0, &east, &north, &up);
            //std::cout << "east: " << east << " north: " << north << std::endl;
            cur_frame.push_back(Data(east, north, frame_nbr));
        }

        

        if (frame_nbr == 0)
        {
            for (auto f : cur_frame)
                trackers.push_back(Tracker(f, initial_age, dt, n_states));
        }
        else
        {
            Track(cur_frame, dt, n_states, initial_age, age_threshold, trackers);
        }

        //std::cout << "There are " << trackers.size() << " trackers" << std::endl;

        for (auto t : trackers)
        {
            for(auto pred_pos: t.pred_list_ )
            {
                
                gc.enu2Geodetic(pred_pos.x_, pred_pos.y_, 0, &lat, &lon, &alt);
                //std::cout << "lat: " << lat << " lon: " << lon << std::endl;
                int pix_x, pix_y;
                coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);
                //std::cout << "pix_x: " << pix_x << " pix_y: " << pix_y << std::endl;

                std::vector<cv::Point2f> map_p, camera_p;
                map_p.push_back(cv::Point2f(pix_x, pix_y));

                //transform camera pixel to map pixel
                cv::perspectiveTransform(map_p, camera_p, H.inv());
                //std::cout << "pix_x: " << camera_p[0].x << " pix_y: " << camera_p[0].y << std::endl;

                cv::circle(frame, cv::Point(camera_p[0].x, camera_p[0].y), 3.0, cv::Scalar(255, 0, 0), CV_FILLED, 8, 0);

            }
        }

        frame_nbr++;

        send_client_dummy(coords, coord_i, sock, socket_opened, CAM_IDX);

        if (to_show)
        {
            cv::imshow("detection", frame);
            cv::waitKey(1);
        }
    }

    /*     for (size_t i = 0; i < trackers.size(); i++)
        if (trackers[i].z_list_.size() > 10)
            plotTruthvsPred(trackers[i].z_list_, trackers[i].pred_list_); */

    free(coords);
    free(adfGeoTransform);

    std::cout << "detection end\n";
    return 0;
}