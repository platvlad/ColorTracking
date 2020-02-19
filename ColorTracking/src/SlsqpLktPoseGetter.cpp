#include "SlsqpLktPoseGetter.h"

#include <iostream>
#include <fstream>

#include <boost/make_shared.hpp>
#include <opencv2/calib3d.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "lkt/lmsolver/ParametersFixer.hpp"
#include "lkt/inliers.hpp"

#include <GradientEstimator.h>

glm::mat4 applyResultToPose(const glm::mat4& matr, const double* params);

std::vector<double> rodriguesFromTransform(const glm::mat4& matr)
{
    cv::Matx31f rot_vec = cv::Matx31f();
    cv::Matx33f rot_matr = cv::Matx33f(
        matr[0][0], matr[1][0], matr[2][0],
        matr[0][1], matr[1][1], matr[2][1],
        matr[0][2], matr[1][2], matr[2][2]);
    cv::Rodrigues(rot_matr, rot_vec);
    std::vector<double> result(6);
    result[0] = rot_vec(0, 0);
    result[1] = rot_vec(1, 0);
    result[2] = rot_vec(2, 0);
    result[3] = matr[3][0];
    result[4] = matr[3][1];
    result[5] = matr[3][2];
    return result;
}

SlsqpLktPoseGetter::SlsqpLktPoseGetter(histograms::Object3d* object3d, const glm::mat4& initial_pose, const cv::Mat3b &init_frame) :
    initial_pose(initial_pose),
    num_iterations(12),
    object(object3d),
    feature_info_list(init_frame.size().area()),
    frame(init_frame)
{
    
    cv::cvtColor(init_frame, prev_frame, cv::COLOR_BGR2GRAY);
    const histograms::Mesh& histo_mesh = object3d->getMesh();
    mesh = lkt::Mesh(histo_mesh.getVertices(), histo_mesh.getFaces());
    projection_matrix = object3d->getRenderer().getCameraMatrix();
    feature_info_list.addNewFeatures(prev_frame, mesh, initial_pose, projection_matrix);

    std::vector<double> params = std::vector<double>(6, 0);
    lkt::lm::ParametersFixer fixer(params, std::vector<bool>(params.size(), false));

    pnp_obj = boost::make_shared<lkt::PnPObjective>(
        feature_info_list.getObjectPosVector(),
        feature_info_list.getImagePtsVector(),
        glm::mat4(1.0),
        projection_matrix,
        fixer,
        boost::make_shared<lkt::lm::HuberLoss>());

    float mesh_diameter = histo_mesh.getBBDiameter();
    max_translation_shift = 0.2f * mesh_diameter;
    max_rotation_shift = 0.5f;

    opt = nlopt_create(NLOPT_LD_SLSQP, 6);

    // is it ok to provide this as parameter from constructor?
    nlopt_set_min_objective(opt, energy_function, this);
    nlopt_set_ftol_rel(opt, 1e-5);
    nlopt_set_maxeval(opt, num_iterations);
}

double SlsqpLktPoseGetter::energy_function(unsigned n, const double *x, double *grad, void *my_func_data)
{

    SlsqpLktPoseGetter* passed_data = reinterpret_cast<SlsqpLktPoseGetter*>(my_func_data);
    histograms::Object3d* object = passed_data->object;
    cv::Mat& frame = passed_data->frame;

    int histo_part = 10;

    glm::mat4& initial_pose = passed_data->initial_pose;

    //glm::mat4 transform_matrix = applyResultToPose(glm::mat4(1.0), x);

    glm::mat4 transform_matrix = applyResultToPose(initial_pose, x);

    std::vector<double> rodrigue_params = rodriguesFromTransform(transform_matrix);
    
    int num_features = passed_data->feature_info_list.getFeatureCount();
    int num_measurements = 2 * num_features;

    std::vector<double> feat_errors(num_measurements);
    lkt::lm::VectorViewD err(feat_errors);
    lkt::lm::DenseMatrix jacobian(num_measurements, 6, 0);
    histograms::PoseEstimator estimator;
    float color_err = estimator.estimateEnergy(*object, frame, transform_matrix, histo_part, false);
    passed_data->pnp_obj->computeView(rodrigue_params, err, jacobian, true);
    
    cv::Mat1d err_vector(num_measurements, 1);
    for (int i = 0; i < num_measurements; ++i)
    {
        err_vector(i, 0) = err[i];
    }
    cv::Mat1d err_transposed;
    cv::transpose(err_vector, err_transposed);
    cv::Mat1d err_matr = err_transposed * err_vector;
    double lk_error = err_matr(0, 0) * 4e-5;
    double err_value = lk_error + color_err;
    std::cout << "feat error = " << lk_error << "; color error = " << color_err << std::endl;
    if (grad)
    {
        GradientEstimator::getGradient(initial_pose, estimator, grad);
        //GradientEstimator::getGradient(glm::mat4(1.0), estimator, grad);
        lkt::lm::VectorD grad_vec(6);
        lkt::lm::DenseMatrixTraits::transposeAndMulByVec(jacobian, feat_errors, grad_vec);
        for (int i = 0; i < 6; ++i)
        {
            grad[i] += grad_vec[i] * 5e-7;
        }
    }
    return err_value;
    
}


double SlsqpLktPoseGetter::energy_function_for_plot(const double *x, std::string plot_type)
{
    int histo_part = 10;

    glm::mat4 transform_matrix = applyResultToPose(initial_pose, x);

    std::vector<double> rodrigue_params = rodriguesFromTransform(transform_matrix);

    int num_features = feature_info_list.getFeatureCount();
    int num_measurements = 2 * num_features;

    std::vector<double> feat_errors(num_measurements);
    lkt::lm::VectorViewD err(feat_errors);
    lkt::lm::DenseMatrix jacobian(num_measurements, 6, 0);
    histograms::PoseEstimator estimator;
    float color_err = estimator.estimateEnergy(*object, frame, transform_matrix, histo_part, false);
    pnp_obj->computeView(rodrigue_params, err, jacobian, true);

    cv::Mat1d err_vector(num_measurements, 1);
    for (int i = 0; i < num_measurements; ++i)
    {
        err_vector(i, 0) = err[i];
    }
    cv::Mat1d err_transposed;
    cv::transpose(err_vector, err_transposed);
    cv::Mat1d err_matr = err_transposed * err_vector;
    double lk_error = err_matr(0, 0) * 4e-5;
    double err_value = lk_error + color_err;
    if (plot_type == "mixed")
    {
        return err_value;
    }
    if (plot_type == "color")
    {
        return color_err;
    }
    return lk_error;
}


glm::mat4 SlsqpLktPoseGetter::getPose(const cv::Mat& new_frame, int mode, std::string directory_name, int frame_number)
{
    if (mode != 0)
    {
        std::cout << "Achtung!" << std::endl;
    }
    nlopt_set_maxeval(opt, 400);

    double x[6];
    for (int i = 0; i < 6; ++i)
    {
        x[i] = 0;
    }
    cv::Mat1b gray_frame;
    cv::cvtColor(new_frame, gray_frame, cv::COLOR_BGR2GRAY);

    feature_info_list.moveFeaturesBySparseFlow(prev_frame, gray_frame);

    std::vector<double> params = std::vector<double>(6, 0);
    lkt::lm::ParametersFixer fixer(params, std::vector<bool>(params.size(), false));

    std::vector<glm::vec3> object_poses = feature_info_list.getObjectPosVector();
    std::vector<glm::vec2> image_points = feature_info_list.getImagePtsVector();

    pnp_obj = boost::make_shared<lkt::PnPObjective>(
        object_poses,
        image_points,
        glm::mat4(1.0),
        projection_matrix,
        fixer,
        boost::make_shared<lkt::lm::HuberLoss>());
        
    frame = new_frame;

    double lower_bounds[6] =
    { x[0] - max_rotation_shift, x[1] - max_rotation_shift, x[2] - max_rotation_shift,
      x[3] - max_translation_shift, x[4] - max_translation_shift, x[5] - max_translation_shift };
    double upper_bounds[6] =
    { x[0] + max_rotation_shift, x[1] + max_rotation_shift, x[2] + max_rotation_shift,
      x[3] + max_translation_shift, x[4] + max_translation_shift, x[5] + max_translation_shift };
    nlopt_set_lower_bounds(opt, lower_bounds);
    nlopt_set_upper_bounds(opt, upper_bounds);
    double minf;

    nlopt_result status = nlopt_optimize(opt, x, &minf);

    std::cout << minf << std::endl;

    prev_frame = gray_frame;

    if (status >= 0)
    {
        initial_pose = applyResultToPose(initial_pose, x);
        if (directory_name != "")
        {
            plotEnergy(initial_pose, directory_name, frame_number);
        }
        float maxInlierError = lkt::computeMaxPnPErr(frame.cols, frame.rows);

        glm::mat4 mvp = projection_matrix * initial_pose;
        feature_info_list.filterOutliers(mvp, maxInlierError);

        feature_info_list.addNewFeatures(gray_frame, mesh, initial_pose, projection_matrix);
        return initial_pose;
    }
    return glm::mat4();
}

glm::mat4 SlsqpLktPoseGetter::getPose(const cv::Mat &frame)
{
    return getPose(frame, 0, "", 0);
}

glm::mat4 SlsqpLktPoseGetter::getPose(const cv::Mat& frame, int mode)
{
    return getPose(frame, mode, "", 0);
}

void SlsqpLktPoseGetter::setInitialPose(const glm::mat4 &pose)
{
    initial_pose = pose;
}

void SlsqpLktPoseGetter::plotAxis(
    const std::string &file_name, 
    const std::vector<float> &axis_values, 
    const glm::mat4 &pose,
    size_t axis_number)
{
    std::ofstream fout_rot(file_name);
    fout_rot << "frames:" << std::endl;
    double x[6] = { 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < axis_values.size(); ++i)
    {
        x[axis_number] = axis_values[i];
        fout_rot << "  - frame: " << i + 1 << std::endl;
        fout_rot << "    error: " << energy_function_for_plot(x, "lkt") << std::endl;
    }
    fout_rot.close();
}

void SlsqpLktPoseGetter::plotEnergy(const glm::mat4 &pose, const std::string &directory_name, int frame_number)
{
    int num_points = 100;
    float max_rotation = 0.1f;
    float max_translation = 0.1f * object->getMesh().getBBDiameter();
    float rotation_step = max_rotation / static_cast<float>(num_points);
    float translation_step = max_translation / static_cast<float>(num_points);
    std::string base_file_name =
        directory_name + "/plots/" + std::to_string(frame_number);
    std::string rot_x_file = base_file_name + "rot_x.yml";
    std::string rot_y_file = base_file_name + "rot_y.yml";
    std::string rot_z_file = base_file_name + "rot_z.yml";
    std::string tr_x_file = base_file_name + "tr_x.yml";
    std::string tr_y_file = base_file_name + "tr_y.yml";
    std::string tr_z_file = base_file_name + "tr_z.yml";
    std::vector<float> angles(2 * num_points + 1);
    std::vector<float> offsets(2 * num_points + 1);
    for (int pt = -num_points; pt <= num_points; ++pt)
    {
        int pose_number = pt + num_points;
        angles[pose_number] = rotation_step * pt;
        offsets[pose_number] = translation_step * pt;
    }
    plotAxis(rot_x_file, angles, pose, 0);
    plotAxis(rot_y_file, angles, pose, 1);
    plotAxis(rot_z_file, angles, pose, 2);
    plotAxis(tr_x_file, offsets, pose, 3);
    plotAxis(tr_y_file, offsets, pose, 4);
    plotAxis(tr_z_file, offsets, pose, 5);
}