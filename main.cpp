#include "ffbpneuralnet.h"
#include "marching_cubes.h"
using namespace marching_cubes;

#include <iostream>
#include <ctime>
#include <random>
using namespace std;


const unsigned int num_components = 4;

vector<double> qmul(const vector<double>& qaqb)
{
	double temp_a_x = qaqb[0];
	double temp_a_y = qaqb[1];
	double temp_a_z = qaqb[2];
	double temp_a_w = qaqb[3];

	double temp_b_x = qaqb[4];
	double temp_b_y = qaqb[5];
	double temp_b_z = qaqb[6];
	double temp_b_w = qaqb[7];

	vector<double> out;

	out.push_back(temp_a_x * temp_b_x - temp_a_y * temp_b_y - temp_a_z * temp_b_z - temp_a_w * temp_b_w);
	out.push_back(temp_a_x * temp_b_y + temp_a_y * temp_b_x + temp_a_z * temp_b_w - temp_a_w * temp_b_z);
	out.push_back(temp_a_x * temp_b_z - temp_a_y * temp_b_w + temp_a_z * temp_b_x + temp_a_w * temp_b_y);
	out.push_back(temp_a_x * temp_b_w + temp_a_y * temp_b_z - temp_a_z * temp_b_y + temp_a_w * temp_b_x);

	return out;
}

vector<double> qmul_ann(const vector<double>& qaqb, FFBPNeuralNet &NNet)
{
	NNet.FeedForward(qaqb);

	vector<double> predicted;
	NNet.GetOutputValues(predicted);

	return predicted;
}

double iterate(const quaternion& src_Z, const quaternion &C, const short unsigned int& max_iterations, const float& threshold, FFBPNeuralNet& NNet, const bool use_network)
{
	quaternion Z = src_Z;

	double len_sq = Z.self_dot();
	const double threshold_sq = threshold * threshold;

	for (short unsigned int i = 0; i < max_iterations; i++)
	{
		vector<double> ZZ_flat(8);
		ZZ_flat[0] = Z.x;
		ZZ_flat[1] = Z.y;
		ZZ_flat[2] = Z.z;
		ZZ_flat[3] = Z.w;
		ZZ_flat[4] = Z.x;
		ZZ_flat[5] = Z.y;
		ZZ_flat[6] = Z.z;
		ZZ_flat[7] = Z.w;

		vector<double> Z_out;

		if(use_network)
			Z_out = qmul_ann(ZZ_flat, NNet);
		else
			Z_out = qmul(ZZ_flat);

		Z.x = Z_out[0] + C.x;
		Z.y = Z_out[1] + C.y;
		Z.z = Z_out[2] + C.z;
		Z.w = Z_out[3] + C.w;

		if ((len_sq = Z.self_dot()) >= threshold_sq)
			break;
	}

	return sqrt(len_sq);
}

bool write_triangles_to_binary_stereo_lithography_file(const vector<triangle>& triangles, const char* const file_name)
{
	cout << "Triangle count: " << triangles.size() << endl;

	if (0 == triangles.size())
		return false;

	// Write to file.
	ofstream out(file_name, ios_base::binary);

	if (out.fail())
		return false;

	const size_t header_size = 80;
	vector<char> buffer(header_size, 0);
	const unsigned int num_triangles = static_cast<unsigned int>(triangles.size()); // Must be 4-byte unsigned int.
	vertex_3 normal;

	// Write blank header.
	out.write(reinterpret_cast<const char*>(&(buffer[0])), header_size);

	// Write number of triangles.
	out.write(reinterpret_cast<const char*>(&num_triangles), sizeof(unsigned int));

	// Copy everything to a single buffer.
	// We do this here because calling ofstream::write() only once PER MESH is going to 
	// send the data to disk faster than if we were to instead call ofstream::write()
	// thirteen times PER TRIANGLE.
	// Of course, the trade-off is that we are using 2x the RAM than what's absolutely required,
	// but the trade-off is often very much worth it (especially so for meshes with millions of triangles).
	cout << "Generating normal/vertex/attribute buffer" << endl;

	// Enough bytes for twelve 4-byte floats plus one 2-byte integer, per triangle.
	const size_t data_size = (12 * sizeof(float) + sizeof(short unsigned int)) * num_triangles;
	buffer.resize(data_size, 0);

	// Use a pointer to assist with the copying.
	// Should probably use std::copy() instead, but memcpy() does the trick, so whatever...
	char* cp = &buffer[0];

	for (vector<triangle>::const_iterator i = triangles.begin(); i != triangles.end(); i++)
	{
		// Get face normal.
		vertex_3 v0 = i->vertex[1] - i->vertex[0];
		vertex_3 v1 = i->vertex[2] - i->vertex[0];
		normal = v0.cross(v1);
		normal.normalize();

		memcpy(cp, &normal.x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &normal.y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &normal.z, sizeof(float)); cp += sizeof(float);

		memcpy(cp, &i->vertex[0].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[0].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[0].z, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[1].z, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].x, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].y, sizeof(float)); cp += sizeof(float);
		memcpy(cp, &i->vertex[2].z, sizeof(float)); cp += sizeof(float);

		cp += sizeof(short unsigned int);
	}

	cout << "Writing " << data_size / 1048576 << " MB of data to binary Stereo Lithography file: " << file_name << endl;

	out.write(reinterpret_cast<const char*>(&buffer[0]), data_size);
	out.close();

	return true;
}

void generate_stl_file(const bool use_network, const float threshold_max, const float threshold_min, const float threshold, const char* const file_name, FFBPNeuralNet &NNet)
{
	const float grid_max = 1.5;
	const float grid_min = -grid_max;
	const size_t res = 500;

	const bool make_border = true;

	const float z_w = 0;
	quaternion C;
	C.x = 0.3f;
	C.y = 0.5f;
	C.z = 0.4f;
	C.w = 0.2f;
	const unsigned short int max_iterations = 8;

	// When adding a border, use a value that is greater than the threshold.
	const float border_value = 1.0f + threshold;
	const double mid_threshold = (threshold_max + threshold_min) * 0.5;

	vector<triangle> triangles;
	vector<float> xyplane0(res * res, 0);
	vector<float> xyplane1(res * res, 0);

	const float step_size = (grid_max - grid_min) / (res - 1);

	size_t z = 0;

	quaternion Z(grid_min, grid_min, grid_min, z_w);

	// Calculate xy plane 0.
	for (size_t x = 0; x < res; x++, Z.x += step_size)
	{
		Z.y = grid_min;

		for (size_t y = 0; y < res; y++, Z.y += step_size)
		{
			if (true == make_border && (x == 0 || y == 0 || z == 0 || x == res - 1 || y == res - 1 || z == res - 1))
				xyplane0[x * res + y] = border_value;
			else
			{
				if (z < res / 2)
				{
					xyplane0[x * res + y] = static_cast<float>(iterate(Z, C, max_iterations, threshold, NNet, use_network));
					xyplane0[x * res + y] = fabsf(xyplane0[x * res + y] - mid_threshold);
				}
				else
					xyplane0[x * res + y] = border_value;
			}
		}
	}

	// Prepare for xy plane 1.
	z++;
	Z.z += step_size;

	size_t box_count = 0;

	// Calculate xy planes 1 and greater.
	for (; z < res; z++, Z.z += step_size)
	{
		Z.x = grid_min;

		cout << "Calculating triangles from xy-plane pair " << z << " of " << res - 1 << endl;

		for (size_t x = 0; x < res; x++, Z.x += step_size)
		{
			Z.y = grid_min;

			for (size_t y = 0; y < res; y++, Z.y += step_size)
			{
				if (true == make_border && (x == 0 || y == 0 || z == 0 || x == res - 1 || y == res - 1 || z == res - 1))
					xyplane1[x * res + y] = border_value;
				else
				{
					if (z < res / 2)
					{
						xyplane1[x * res + y] = static_cast<float>(iterate(Z, C, max_iterations, threshold, NNet, use_network));
						xyplane1[x * res + y] = fabsf(xyplane1[x * res + y] - mid_threshold);
					}
					else
						xyplane1[x * res + y] = border_value;
				}
			}
		}

		// Calculate triangles for the xy-planes corresponding to z - 1 and z by marching cubes.
		tesselate_adjacent_xy_plane_pair(
			box_count,
			xyplane0, xyplane1,
			z - 1,
			triangles,
			threshold_max - mid_threshold, // Use threshold as isovalue.
			grid_min, grid_max, res,
			grid_min, grid_max, res,
			grid_min, grid_max, res);

		// Swap memory pointers (fast) instead of performing a memory copy (slow).
		xyplane1.swap(xyplane0);
	}

	cout << endl;

	if (0 < triangles.size())
		write_triangles_to_binary_stereo_lithography_file(triangles, file_name);

	// Print box-counting dimension
	// Make sure that step_size != 1.0f :)
	cout << "Box counting dimension: " << logf(static_cast<float>(box_count)) / logf(1.0f / step_size) << endl;

}

int main(void)
{
	const float threshold = 4.0;

	std::mt19937 generator_real(static_cast<unsigned>(time(0)));
	std::uniform_real_distribution<float> dis_real(-threshold, threshold);


	// 1) Train network. This section can be commented out after the network
	// has been trained.
	//
	//vector<size_t> HiddenLayers;
	//HiddenLayers.push_back(8 * num_components);
	//HiddenLayers.push_back(16 * num_components);
	//HiddenLayers.push_back(32 * num_components);

	//FFBPNeuralNet NNet(2 * num_components, HiddenLayers, num_components);

	//NNet.SetLearningRate(0.0001);
	//NNet.SetMomentum(0.1);

	///*const double max_error_rate = 0.00001;*/
	//const long unsigned int max_training_sessions = 1000000000;

	//double error_rate = 0.0;
	//long unsigned int num_training_sessions = 0;

	//do
	//{
	//	if (num_training_sessions % 1000 == 0)
	//		cout << num_training_sessions / static_cast<float>(max_training_sessions) << endl;

	//	vector<double> io;

	//	for (size_t i = 0; i < 2 * num_components; i++)
	//		io.push_back(dis_real(generator_real));

	//	NNet.FeedForward(io);
	//	io = qmul(io);
	//	error_rate = NNet.BackPropagate(io);
	//	error_rate = sqrt(error_rate);

	//	num_training_sessions++;

	//} while (/*error_rate >= max_error_rate &&*/ num_training_sessions < max_training_sessions);

	//cout << "Final number of training sessions/epochs: " << num_training_sessions << endl;
	//cout << "Final error rate: " << error_rate << endl;
	//cout << endl;

	//NNet.SaveToFile("network.bin");






	// 2) Load a network from a file
	//
	FFBPNeuralNet NNet2("network.bin");


	// 3) Generate isosurfaces
	//
	generate_stl_file(false, threshold * 0.25 - 0.1, 0, threshold, "out0.stl", NNet2);
	generate_stl_file(false, threshold * 0.5 - 0.1, threshold * 0.25, threshold, "out1.stl", NNet2);
	generate_stl_file(false, threshold, threshold * 0.5, threshold, "out2.stl", NNet2);
	generate_stl_file(true, threshold * 0.25 - 0.1, 0, threshold, "out0_network.stl", NNet2);
	generate_stl_file(true, threshold * 0.5 - 0.1, threshold * 0.25, threshold, "out1_network.stl", NNet2);
	generate_stl_file(true, threshold, threshold * 0.5, threshold, "out2_network.stl", NNet2);

	return 0;
}