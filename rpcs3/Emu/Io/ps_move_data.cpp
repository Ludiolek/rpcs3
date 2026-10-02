#include "stdafx.h"
#include "ps_move_data.h"
#include "Utilities/File.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

LOG_CHANNEL(move_log, "Move");

const ps_move_data::vect<4> ps_move_data::default_quaternion = ps_move_data::vect<4>({ 0.0f, 0.0f, 0.0f, 1.0f });

ps_move_data::ps_move_data()
	: quaternion(default_quaternion)
{
}

void ps_move_data::reset_sensors()
{
	quaternion = default_quaternion;
	accelerometer = {};
	gyro = {};
	prev_gyro = {};
	angular_acceleration = {};
	magnetometer = {};
	//prev_pos_world = {}; // probably no reset needed ?
	vel_world = {};
	prev_vel_world = {};
	accel_world = {};
	angvel_world = {};
	angaccel_world = {};
}

ps_move_data::vect<3> ps_move_data::rotate_vector(const vect<4>& q, const vect<3>& v)
{
	const auto cross = [](const vect<3>& a, const vect<3>& b)
	{
		return vect<3>({
			a.y() * b.z() - a.z() * b.y(),
			a.z() * b.x() - a.x() * b.z(),
			a.x() * b.y() - a.y() * b.x()
		});
	};

	// q = (x, y, z, w)
	const vect<3> q_vec({q.x(), q.y(), q.z()});

	// t = 2 * cross(q_vec, v)
	const vect<3> t = cross(q_vec, v) * 2.0f;

	// v' = v + w * t + cross(q_vec, t)
	const vect<3> v_prime = v + t * q.w() + cross(q_vec, t);

	return v_prime;
}

void ps_move_data::update_orientation(f32 delta_time)
{
	if (!delta_time)
		return;

	if constexpr (use_imu_for_velocity)
	{
		// Gravity in world frame
		constexpr f32 gravity = 9.81f;
		constexpr vect<3> g({0.0f, 0.0f, -gravity});

		// Rotate gravity into sensor frame
		const vect<3> g_sensor = rotate_vector(quaternion, g);

		// Remove gravity
		vect<3> linear_local;
		for (u32 i = 0; i < 3; i++)
		{
			linear_local[i] = (accelerometer[i] * gravity) - g_sensor[i];
		}

		// Linear acceleration (rotate to world coordinates)
		accel_world = rotate_vector(quaternion, linear_local);

		// convert to mm/s²
		for (u32 i = 0; i < 3; i++)
		{
			accel_world[i] *= 1000.0f;
		}

		// Linear velocity (integrate acceleration)
		for (u32 i = 0; i < 3; i++)
		{
			vel_world[i] = prev_vel_world[i] + accel_world[i] * delta_time;
		}

		prev_vel_world = vel_world;
	}

	// Compute raw angular acceleration
	for (u32 i = 0; i < 3; i++)
	{
		const f32 alpha = (gyro[i] - prev_gyro[i]) / delta_time;

		// Filtering
		constexpr f32 weight = 0.8f;
		constexpr f32 weight_inv = 1.0f - weight;
		angular_acceleration[i] = weight * angular_acceleration[i] + weight_inv * alpha;
	}

	// Angular velocity (rotate to world coordinates)
	angvel_world = rotate_vector(quaternion, gyro);

	// Angular acceleration (rotate to world coordinates)
	angaccel_world = rotate_vector(quaternion, angular_acceleration);

	prev_gyro = gyro;
}

// ---------------------------------------------------------------------------------------------
// PS Move position filter (patched build)
//
// The position of the sphere comes from the camera image. The distance to the camera is derived
// from the radius of the sphere in pixels, which is only a handful of pixels at living room
// distances. A fraction of a pixel of noise in the radius moves the reported position by many
// centimeters within a single frame, which games see as a fast movement of the controller.
// This filter smooths the position (strongly while it is nearly static, weakly while it really
// moves) and derives velocity and acceleration from the smoothed position.
// All parameters can be changed at runtime in config/ps_move_tuning.txt (keys start with pos_).
// ---------------------------------------------------------------------------------------------
namespace
{
	struct pos_filter_tuning
	{
		f32 enable = 1.0f;          // 0 = original behavior
		f32 xy_min_cutoff = 1.5f;   // Hz. Lower = smoother but more lag while nearly still
		f32 xy_beta = 0.01f;        // Hz per mm/s. Higher = less lag during fast movement
		f32 z_min_cutoff = 0.15f;   // Hz. The distance is much noisier than x/y
		f32 z_beta = 0.0005f;       // Hz per mm/s
		f32 d_cutoff = 1.0f;        // Hz. Smoothing of the speed estimate that drives the adaptive cutoff
		f32 z_median = 7.0f;        // Number of frames for the median filter on the distance (1 = off, max 9)
		f32 vel_cutoff = 4.0f;      // Hz. Smoothing of the reported velocity
		f32 accel_cutoff = 3.0f;    // Hz. Smoothing of the reported acceleration
		f32 max_gap = 0.5f;         // Seconds without a frame after which the filter restarts
		f32 log_enable = 1.0f;
		f32 log_interval = 1.0f;
	};

	struct pos_one_euro
	{
		f32 value = 0.0f;
		f32 speed = 0.0f;

		static f32 alpha(f32 cutoff_hz, f32 dt)
		{
			const f32 r = 2.0f * 3.14159265f * std::max(cutoff_hz, 0.001f) * dt;
			return r / (r + 1.0f);
		}

		void reset(f32 raw)
		{
			value = raw;
			speed = 0.0f;
		}

		f32 update(f32 raw, f32 dt, f32 min_cutoff, f32 beta, f32 d_cutoff)
		{
			const f32 raw_speed = (raw - value) / dt;
			speed += alpha(d_cutoff, dt) * (raw_speed - speed);
			const f32 cutoff = min_cutoff + beta * std::abs(speed);
			value += alpha(cutoff, dt) * (raw - value);
			return value;
		}
	};

	struct pos_filter_state
	{
		bool valid = false;
		int index = 0;
		std::array<pos_one_euro, 3> filters {};
		std::array<f32, 3> pos {};
		std::array<f32, 3> raw_prev {};
		std::array<f32, 9> z_history {};
		u32 z_count = 0;

		// Statistics for the log
		u64 last_log_time_us = 0;
		u32 frames = 0;
		std::array<f32, 3> raw_min {}, raw_max {}, flt_min {}, flt_max {};
		f32 raw_speed_max = 0.0f;
		f32 flt_speed_max = 0.0f;

		void reset_stats()
		{
			frames = 0;
			raw_speed_max = 0.0f;
			flt_speed_max = 0.0f;
		}
	};

	std::mutex s_pos_mutex;
	std::unordered_map<const ps_move_data*, pos_filter_state> s_pos_states;
	pos_filter_tuning s_pos_tuning {};
	std::string s_pos_tuning_text;
	u64 s_pos_tuning_time_us = 0;
	bool s_pos_tuning_loaded = false;

	void pos_parse_tuning(const std::string& text, pos_filter_tuning& tuning)
	{
		const std::pair<std::string_view, f32*> keys[] =
		{
			{ "pos_filter_enable", &tuning.enable },
			{ "pos_xy_min_cutoff", &tuning.xy_min_cutoff },
			{ "pos_xy_beta", &tuning.xy_beta },
			{ "pos_z_min_cutoff", &tuning.z_min_cutoff },
			{ "pos_z_beta", &tuning.z_beta },
			{ "pos_d_cutoff", &tuning.d_cutoff },
			{ "pos_z_median", &tuning.z_median },
			{ "pos_vel_cutoff", &tuning.vel_cutoff },
			{ "pos_accel_cutoff", &tuning.accel_cutoff },
			{ "pos_max_gap", &tuning.max_gap },
			{ "pos_log_enable", &tuning.log_enable },
			{ "pos_log_interval", &tuning.log_interval },
		};

		const auto trim = [](std::string_view s)
		{
			while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.remove_prefix(1);
			while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
			return s;
		};

		usz start = 0;

		while (start < text.size())
		{
			usz end = text.find('\n', start);
			if (end == std::string::npos) end = text.size();

			std::string_view line(text.data() + start, end - start);
			start = end + 1;

			if (const usz comment = line.find('#'); comment != std::string_view::npos)
			{
				line = line.substr(0, comment);
			}

			const usz eq = line.find('=');
			if (eq == std::string_view::npos) continue;

			const std::string_view key = trim(line.substr(0, eq));
			const std::string value(trim(line.substr(eq + 1)));
			if (key.empty() || value.empty()) continue;

			for (const auto& [name, ptr] : keys)
			{
				if (key == name)
				{
					char* end_ptr = nullptr;
					const f32 parsed = std::strtof(value.c_str(), &end_ptr);
					if (end_ptr != value.c_str() && std::isfinite(parsed))
					{
						*ptr = parsed;
					}
					break;
				}
			}
		}

		tuning.xy_min_cutoff = std::clamp(tuning.xy_min_cutoff, 0.01f, 1000.0f);
		tuning.z_min_cutoff = std::clamp(tuning.z_min_cutoff, 0.01f, 1000.0f);
		tuning.xy_beta = std::clamp(tuning.xy_beta, 0.0f, 10.0f);
		tuning.z_beta = std::clamp(tuning.z_beta, 0.0f, 10.0f);
		tuning.d_cutoff = std::clamp(tuning.d_cutoff, 0.01f, 1000.0f);
		tuning.z_median = std::clamp(tuning.z_median, 1.0f, 9.0f);
		tuning.vel_cutoff = std::clamp(tuning.vel_cutoff, 0.01f, 1000.0f);
		tuning.accel_cutoff = std::clamp(tuning.accel_cutoff, 0.01f, 1000.0f);
		tuning.max_gap = std::clamp(tuning.max_gap, 0.05f, 10.0f);
		tuning.log_interval = std::clamp(tuning.log_interval, 0.1f, 60.0f);
	}

	// Re-reads the tuning file at most once per second (of wall clock time)
	void pos_update_tuning()
	{
		const u64 now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());

		if (s_pos_tuning_loaded && (now_us - s_pos_tuning_time_us) < 1'000'000)
		{
			return;
		}

		s_pos_tuning_time_us = now_us;

		std::string text;

		if (fs::file file{ fs::get_config_dir(true) + "ps_move_tuning.txt", fs::read })
		{
			text = file.to_string();
		}

		if (s_pos_tuning_loaded && text == s_pos_tuning_text)
		{
			return;
		}

		s_pos_tuning_text = text;
		s_pos_tuning_loaded = true;

		pos_filter_tuning tuning {};
		pos_parse_tuning(text, tuning);
		s_pos_tuning = tuning;

		move_log.notice("MoveTrack: tuning loaded: enable=%.0f, xy_min_cutoff=%.3f, xy_beta=%.4f, z_min_cutoff=%.3f, z_beta=%.4f, d_cutoff=%.2f, z_median=%.0f, vel_cutoff=%.2f, accel_cutoff=%.2f, max_gap=%.2f",
			tuning.enable, tuning.xy_min_cutoff, tuning.xy_beta, tuning.z_min_cutoff, tuning.z_beta, tuning.d_cutoff, tuning.z_median, tuning.vel_cutoff, tuning.accel_cutoff, tuning.max_gap);
	}
}

void ps_move_data::update_velocity(u64 timestamp, be_t<f32> pos_world[4])
{
	if constexpr (use_imu_for_velocity)
		return;

	std::lock_guard lock(s_pos_mutex);

	pos_update_tuning();

	const pos_filter_tuning& tuning = s_pos_tuning;
	const bool filter_enabled = tuning.enable > 0.0f;

	pos_filter_state& state = s_pos_states[this];

	if (state.index == 0)
	{
		state.index = static_cast<int>(s_pos_states.size());
	}

	if (last_velocity_update_time_us == timestamp)
	{
		// Same camera frame as before. Report the same filtered position again.
		if (filter_enabled && state.valid)
		{
			for (u32 i = 0; i < 3; i++)
			{
				pos_world[i] = state.pos[i];
			}
		}
		return;
	}

	// Get elapsed time since last update
	const f32 delta_time = (last_velocity_update_time_us == 0 || timestamp < last_velocity_update_time_us) ? 0.0f : ((timestamp - last_velocity_update_time_us) / 1'000'000.0f);
	last_velocity_update_time_us = timestamp;

	if (!filter_enabled)
	{
		// Original behavior
		state.valid = false;

		if (!delta_time)
			return;

		for (u32 i = 0; i < 3; i++)
		{
			// Linear velocity
			constexpr f32 weight = 0.8f;
			constexpr f32 weight_inv = 1.0f - weight;
			vel_world[i] = weight * ((pos_world[i] - prev_pos_world[i]) / delta_time) + weight_inv * prev_vel_world[i];
			prev_pos_world[i] = pos_world[i];

			// Linear acceleration
			accel_world[i] = (vel_world[i] - prev_vel_world[i]) / delta_time;
		}

		prev_vel_world = vel_world;
		return;
	}

	const std::array<f32, 3> raw = { static_cast<f32>(pos_world[0]), static_cast<f32>(pos_world[1]), static_cast<f32>(pos_world[2]) };
	const bool raw_ok = std::isfinite(raw[0]) && std::isfinite(raw[1]) && std::isfinite(raw[2]);

	if (!raw_ok)
	{
		return;
	}

	if (!state.valid || delta_time <= 0.0f || delta_time > tuning.max_gap)
	{
		// (Re)start the filter at the raw position
		for (u32 i = 0; i < 3; i++)
		{
			state.filters[i].reset(raw[i]);
			state.pos[i] = raw[i];
			state.raw_prev[i] = raw[i];
			prev_pos_world[i] = raw[i];
			vel_world[i] = 0.0f;
			prev_vel_world[i] = 0.0f;
			accel_world[i] = 0.0f;
		}

		state.z_count = 0;
		state.valid = true;
		state.reset_stats();
		state.last_log_time_us = timestamp;
		return;
	}

	// Median filter on the distance to remove single frame spikes
	const u32 median_size = static_cast<u32>(tuning.z_median);

	for (usz i = state.z_history.size() - 1; i > 0; i--)
	{
		state.z_history[i] = state.z_history[i - 1];
	}

	state.z_history[0] = raw[2];
	state.z_count = std::min<u32>(state.z_count + 1, static_cast<u32>(state.z_history.size()));

	const u32 median_count = std::max<u32>(1, std::min(median_size, state.z_count));
	std::array<f32, 9> sorted {};
	std::copy_n(state.z_history.begin(), median_count, sorted.begin());
	std::sort(sorted.begin(), sorted.begin() + median_count);
	const f32 z_median = (median_count % 2) ? sorted[median_count / 2] : (sorted[median_count / 2 - 1] + sorted[median_count / 2]) * 0.5f;

	// Smooth the distance
	const f32 z = state.filters[2].update(z_median, delta_time, tuning.z_min_cutoff, tuning.z_beta, tuning.d_cutoff);

	// x and y were scaled with the same noisy radius as the distance. Rescale them with the smoothed distance.
	const f32 scale = (raw[2] > 1.0f) ? std::clamp(z / raw[2], 0.25f, 4.0f) : 1.0f;
	const f32 x = state.filters[0].update(raw[0] * scale, delta_time, tuning.xy_min_cutoff, tuning.xy_beta, tuning.d_cutoff);
	const f32 y = state.filters[1].update(raw[1] * scale, delta_time, tuning.xy_min_cutoff, tuning.xy_beta, tuning.d_cutoff);

	const std::array<f32, 3> filtered = { x, y, z };

	const f32 vel_alpha = pos_one_euro::alpha(tuning.vel_cutoff, delta_time);
	const f32 accel_alpha = pos_one_euro::alpha(tuning.accel_cutoff, delta_time);

	f32 raw_speed_sq = 0.0f;
	f32 flt_speed_sq = 0.0f;

	for (u32 i = 0; i < 3; i++)
	{
		// Linear velocity
		const f32 vel = (filtered[i] - state.pos[i]) / delta_time;
		vel_world[i] = prev_vel_world[i] + vel_alpha * (vel - prev_vel_world[i]);

		// Linear acceleration
		const f32 accel = (vel_world[i] - prev_vel_world[i]) / delta_time;
		accel_world[i] += accel_alpha * (accel - accel_world[i]);

		const f32 raw_vel = (raw[i] - state.raw_prev[i]) / delta_time;
		raw_speed_sq += raw_vel * raw_vel;
		flt_speed_sq += vel_world[i] * vel_world[i];

		if (state.frames == 0)
		{
			state.raw_min[i] = state.raw_max[i] = raw[i];
			state.flt_min[i] = state.flt_max[i] = filtered[i];
		}
		else
		{
			state.raw_min[i] = std::min(state.raw_min[i], raw[i]);
			state.raw_max[i] = std::max(state.raw_max[i], raw[i]);
			state.flt_min[i] = std::min(state.flt_min[i], filtered[i]);
			state.flt_max[i] = std::max(state.flt_max[i], filtered[i]);
		}

		state.pos[i] = filtered[i];
		state.raw_prev[i] = raw[i];
		prev_pos_world[i] = filtered[i];
		pos_world[i] = filtered[i];
	}

	prev_vel_world = vel_world;

	state.raw_speed_max = std::max(state.raw_speed_max, std::sqrt(raw_speed_sq));
	state.flt_speed_max = std::max(state.flt_speed_max, std::sqrt(flt_speed_sq));
	state.frames++;

	if (tuning.log_enable > 0.0f && (timestamp - state.last_log_time_us) >= static_cast<u64>(tuning.log_interval * 1'000'000.0f))
	{
		move_log.notice("MoveTrack: id=%d, frames=%d, raw x=[%.0f, %.0f] y=[%.0f, %.0f] z=[%.0f, %.0f], filtered x=[%.0f, %.0f] y=[%.0f, %.0f] z=[%.0f, %.0f], max speed raw=%.0f filtered=%.0f mm/s",
			state.index, state.frames,
			state.raw_min[0], state.raw_max[0], state.raw_min[1], state.raw_max[1], state.raw_min[2], state.raw_max[2],
			state.flt_min[0], state.flt_max[0], state.flt_min[1], state.flt_max[1], state.flt_min[2], state.flt_max[2],
			state.raw_speed_max, state.flt_speed_max);

		state.reset_stats();
		state.last_log_time_us = timestamp;
	}
}
