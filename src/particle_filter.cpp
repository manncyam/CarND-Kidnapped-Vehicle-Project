/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>

#include "particle_filter.h"

using namespace std;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 200;
  weights.resize(num_particles, 1.0f);
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  // init all particles
  for(size_t i = 0; i < num_particles; ++i)
  {	
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.id = i;
    p.weight = 1.0f;
    particles.push_back(p);	
  }
  is_initialized = true;  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_yaw(0, std_pos[2]);
  
  for(auto particle = particles.begin(); particle != particles.end(); ++particle)
  {
	// check to prevent divided by 0
	if(std::fabs(yaw_rate) > 0.01)
    {
      particle->x += velocity / yaw_rate * (sin(particle->theta + yaw_rate*delta_t) - sin(particle->theta));
	  particle->y += velocity / yaw_rate * (cos(particle->theta) - cos(particle->theta + yaw_rate * delta_t));
    }else{
	  particle->x += velocity * delta_t * cos(particle->theta);
	  particle->y += velocity * delta_t * sin(particle->theta);   
	}
    particle->theta += yaw_rate * delta_t;	
	// add gaussian noise
    particle->x += dist_x(gen);
	particle->y += dist_y(gen);
	particle->theta += dist_yaw(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  // calculate the data association by finding the shortest distance
  for(auto obs = observations.begin(); obs != observations.end(); ++obs)
  {
    double min_distance{INFINITY}, dist, dx, dy;
    int minIdx{-1}, i{0};
    for (auto pred = predicted.begin(); pred != predicted.end(); ++pred)	
    {
	  dx = pred->x - obs->x;
      dy = pred->y - obs->y;
      dist = dx * dx + dy * dy;
      if(dist < min_distance)
	  {
	    min_distance = dist;
		minIdx = i;
      }		  
	  i++;
    }
    obs->id = minIdx;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  auto cov_x = std_landmark[0] * std_landmark[0];
  auto cov_y = std_landmark[1] * std_landmark[1];
  auto normalizer = 2.0 * M_PI * std_landmark[0] * std_landmark[1];	
  
  for(size_t i = 0; i < particles.size(); ++i)
  {
    auto p = particles[i];
	// stored predicted landmarks
    std::vector<LandmarkObs> predicted_landmarks;
    for(auto lm : map_landmarks.landmark_list)
    {
      LandmarkObs lm_pred;
      lm_pred.x = lm.x_f;
      lm_pred.y = lm.y_f;
      lm_pred.id = lm.id_i;
      auto dx = lm_pred.x - p.x;
      auto dy = lm_pred.y - p.y;

      // is within the range?
      if( (dx * dx + dy * dy) <= sensor_range * sensor_range){
        predicted_landmarks.push_back(lm_pred);
	  }
    } /*for(auto lm : map_landmarks.landmark_list)*/
	
    // stored transformed observation landmarks
    std::vector<LandmarkObs> transformed_obs;
    
	// transform to global coordinate
    for(auto obs_lm : observations)
    {	
      LandmarkObs obs_global;
	  obs_global.x = p.x + obs_lm.x * cos(p.theta) - obs_lm.y * sin(p.theta);
      obs_global.y = p.y + obs_lm.x * sin(p.theta) + obs_lm.y * cos(p.theta);
      obs_global.id = obs_lm.id;
      transformed_obs.push_back(obs_global);
    }/*for(auto obs_lm : observations)*/
	if(predicted_landmarks.size() > 0)
	{
		// associate predicted landmarks to observed landmarks
		dataAssociation(predicted_landmarks, transformed_obs);

		double total_prob = 1.0f;
		// calculate total probability for each particle
		for(auto obs = transformed_obs.begin(); obs != transformed_obs.end(); ++obs)
		{
		  auto assoc_lm = predicted_landmarks[obs->id];
		  auto dx = (obs->x - assoc_lm.x);
		  auto dy = (obs->y - assoc_lm.y);
		  total_prob *= exp(-(dx * dx / (2 * cov_x) + dy * dy / (2 * cov_y))) / normalizer;
		}/*for(auto obs = transformed_obs.begin(); obs != transformed_obs.end(); ++obs)*/
		
		particles[i].weight = total_prob;
		weights[i] = total_prob;
	}
  }/*for(size_t i = 0; i < particles.size(); ++i)*/
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::discrete_distribution<int> dd(weights.begin(), weights.end());
  
  std::vector<Particle> new_particles;
  for(int i = 0; i < num_particles; i++)
  {
    auto indx = dd(gen);
    new_particles.push_back(std::move(particles[indx]));
  }
  // move new_particles to particles
  particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
