// online2bin/online2-tcp-nnet3-decode-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)
//           2018  Polish-Japanese Academy of Information Technology (Author: Danijel Korzinek)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <poll.h>
#include <signal.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <list>

namespace kaldi {

class TcpServer {
 public:
  explicit TcpServer();
  ~TcpServer();

  bool Listen(int32 port);  // start listening on a given port
  int32 Accept();  // accept a client and return its descriptor

 private:
	struct ::sockaddr_in h_addr_;
	int32 server_desc_;
};

class DecodingThread {
 public:
	explicit DecodingThread(int32 client_desc, int read_timeout, OnlineNnet2FeaturePipelineInfo *feature_info, LatticeFasterDecoderConfig *decoder_opts,
		TransitionModel *trans_model, nnet3::DecodableNnetSimpleLoopedInfo *decodable_info, fst::Fst<fst::StdArc> *decode_fst,
		nnet3::NnetSimpleLoopedComputationOptions *decodable_opts, fst::SymbolTable *word_syms, BaseFloat frame_shift,
		int32 frame_subsampling, OnlineEndpointConfig *endpoint_opts, BaseFloat chunk_length_secs, BaseFloat output_period,
		BaseFloat samp_freq);
	~DecodingThread();
	
	void StartDecoding();

	bool ReadChunk(size_t len); // get more data and return false if end-of-stream
	std::string ReadKey(size_t len);
	bool IsConnected();

	Vector<BaseFloat> GetChunk(); // get the data read by above method

	bool Write(const std::string &msg); // write to accepted client
	bool WriteLn(const std::string &msg, const std::string &eol = "\n"); // write line to accepted client

	void Disconnect();

 private:
	struct ::sockaddr_in h_addr_;
	int32 server_desc_, client_desc_;
	int16 *samp_buf_;
	size_t buf_len_, has_read_;
	pollfd client_set_[1];
	int read_timeout_;

	BaseFloat chunk_length_secs = 0.18;
	BaseFloat output_period = 1;
	BaseFloat samp_freq = 8000.0;

	OnlineNnet2FeaturePipelineInfo *feature_info;
	LatticeFasterDecoderConfig *decoder_opts;
	TransitionModel *trans_model;
	nnet3::DecodableNnetSimpleLoopedInfo *decodable_info;
	fst::Fst<fst::StdArc> *decode_fst;
	nnet3::NnetSimpleLoopedComputationOptions *decodable_opts;
	fst::SymbolTable *word_syms;
	BaseFloat frame_shift;
	int32 frame_subsampling;
	OnlineEndpointConfig *endpoint_opts;	
};

std::string LatticeToString(const Lattice &lat, const fst::SymbolTable &word_syms) {
  LatticeWeight weight;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(lat, &alignment, &words, &weight);

  std::ostringstream msg;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms.Find(words[i]);
    if (s.empty()) {
      KALDI_WARN << "Word-id " << words[i] << " not in symbol table.";
      msg << "<#" << std::to_string(i) << "> ";
    } else
      msg << s << " ";
  }
  return msg.str();
}

std::string GetTimeString(int32 t_beg, int32 t_end, BaseFloat time_unit) {
  char buffer[100];
  double t_beg2 = t_beg * time_unit;
  double t_end2 = t_end * time_unit;
  snprintf(buffer, 100, "%.2f %.2f", t_beg2, t_end2);
  return std::string(buffer);
}

int32 GetLatticeTimeSpan(const Lattice& lat) {
  std::vector<int32> times;
  LatticeStateTimes(lat, &times);
  return times.back();
}

std::string LatticeToString(const CompactLattice &clat, const fst::SymbolTable &word_syms) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return "";
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);
  return LatticeToString(best_path_lat, word_syms);
}
}


void Work(kaldi::DecodingThread *dec)
{
	dec->StartDecoding();
}

static bool RemoveInactive(kaldi::DecodingThread *item)
{
	if (!item->IsConnected())
	{
		delete item;
		return true;
	}
	else {
		return false;
	}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in audio from a network socket and performs online\n"
        "decoding with neural nets (nnet3 setup), with iVector-based\n"
        "speaker adaptation and endpointing.\n"
        "Note: some configuration values and inputs are set via config\n"
        "files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-tcp-nnet3-decode-faster [options] <nnet3-in> "
        "<fst-in> <word-symbol-table>\n";

    ParseOptions po(usage);


    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;

    BaseFloat chunk_length_secs = 0.18;
    BaseFloat output_period = 1;
    BaseFloat samp_freq = 16000.0;
    int port_num = 5050;
    int read_timeout = 3;
    bool produce_time = false;
	std::string access_key = "";

    po.Register("samp-freq", &samp_freq,
                "Sampling frequency of the input signal (coded as 16-bit slinear).");
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.");
    po.Register("output-period", &output_period,
                "How often in seconds, do we check for changes in output.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");
    po.Register("read-timeout", &read_timeout,
                "Number of seconds of timout for TCP audio data to appear on the stream. Use -1 for blocking.");
    po.Register("port-num", &port_num,
                "Port number the server will listen on.");
    po.Register("produce-time", &produce_time,
                "Prepend begin/end times between endpoints (e.g. '5.46 6.81 <text_output>', in seconds)");

	po.Register("access-key", &access_key,
		"Access key.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        word_syms_filename = po.GetArg(3);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    BaseFloat frame_shift = feature_info.FrameShiftInSeconds();
    int32 frame_subsampling = decodable_opts.frame_subsampling_factor;

    KALDI_VLOG(1) << "Loading AM...";

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);

    KALDI_VLOG(1) << "Loading FST...";

    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);
	fst::SymbolTable *word_syms = NULL;
    if (!word_syms_filename.empty())
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;

    signal(SIGPIPE, SIG_IGN); // ignore SIGPIPE to avoid crashing when socket forcefully disconnected

    TcpServer server;

    server.Listen(port_num);

	std::list<DecodingThread*> decoders;

    while (true) {

		int32 client = server.Accept();	

		DecodingThread *dec = new DecodingThread(client, read_timeout, &feature_info, &decoder_opts, &trans_model, &decodable_info, decode_fst, &decodable_opts,
			word_syms, frame_shift, frame_subsampling, &endpoint_opts, chunk_length_secs, output_period, samp_freq);
		
		std::string key = dec->ReadKey(access_key.size());
		
		if (access_key.compare(key) != 0)
		{
			dec->Disconnect();
			delete dec;
		}
		else {
			decoders.push_back(dec);
			std::thread thread(Work, dec);
			thread.detach();
		}

		decoders.remove_if(RemoveInactive);

    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
} // main()


namespace kaldi {

	TcpServer::TcpServer() {
		server_desc_ = -1;
	}

	bool TcpServer::Listen(int32 port) {
		h_addr_.sin_addr.s_addr = INADDR_ANY;
		h_addr_.sin_port = htons(port);
		h_addr_.sin_family = AF_INET;

		server_desc_ = socket(AF_INET, SOCK_STREAM, 0);

		if (server_desc_ == -1) {
			KALDI_ERR << "Cannot create TCP socket!";
			return false;
		}

		int32 flag = 1;
		int32 len = sizeof(int32);
		if (setsockopt(server_desc_, SOL_SOCKET, SO_REUSEADDR, &flag, len) == -1) {
			KALDI_ERR << "Cannot set socket options!";
			return false;
		}

		if (bind(server_desc_, (struct sockaddr *) &h_addr_, sizeof(h_addr_)) == -1) {
			KALDI_ERR << "Cannot bind to port: " << port << " (is it taken?)";
			return false;
		}

		if (listen(server_desc_, 1) == -1) {
			KALDI_ERR << "Cannot listen on port!";
			return false;
		}

		KALDI_LOG << "TcpServer: Listening on port: " << port;

		return true;

	}

	TcpServer::~TcpServer() {
		// TO DO. Disconnect All;
		if (server_desc_ != -1)
			close(server_desc_);
	}

	int32 TcpServer::Accept() {
		KALDI_LOG << "Waiting for client...";

		socklen_t len;

		len = sizeof(struct sockaddr);
		int32 client_desc_ = accept(server_desc_, (struct sockaddr *) &h_addr_, &len);

		struct sockaddr_storage addr;
		char ipstr[20];

		len = sizeof addr;
		getpeername(client_desc_, (struct sockaddr *) &addr, &len);

		struct sockaddr_in *s = (struct sockaddr_in *) &addr;
		inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof ipstr);

		KALDI_LOG << "Accepted connection from: " << ipstr;

		return client_desc_;
	}

	DecodingThread::DecodingThread(int32 client_desc, int read_timeout, OnlineNnet2FeaturePipelineInfo *feature_info_, LatticeFasterDecoderConfig *decoder_opts_,
		TransitionModel *trans_model_, nnet3::DecodableNnetSimpleLoopedInfo *decodable_info_, fst::Fst<fst::StdArc> *decode_fst_,
		nnet3::NnetSimpleLoopedComputationOptions *decodable_opts_, fst::SymbolTable *word_syms_, BaseFloat frame_shift_,
		int32 frame_subsampling_, OnlineEndpointConfig *endpoint_opts_, BaseFloat chunk_length_secs_, BaseFloat output_period_,
		BaseFloat samp_freq_) 
	{
		client_desc_ = client_desc;
		client_set_[0].fd = client_desc_;
		client_set_[0].events = POLLIN;

		samp_buf_ = NULL;
		buf_len_ = 0;
		read_timeout_ = 1000 * read_timeout;

		feature_info = feature_info_;
		decoder_opts = decoder_opts_;
		trans_model = trans_model_;
		decodable_info = decodable_info_;
		decode_fst = decode_fst_;
		decodable_opts = decodable_opts_;
		word_syms = word_syms_;
		frame_shift = frame_shift_;
		frame_subsampling = frame_subsampling_;
		endpoint_opts = endpoint_opts_;
		chunk_length_secs = chunk_length_secs_;
		output_period = output_period_;
		samp_freq = samp_freq_;
	}

	bool DecodingThread::IsConnected()
	{
		return client_desc_ != -1;
	}

	void DecodingThread::StartDecoding()
	{

		int32 samp_count = 0;// this is used for output refresh rate
		size_t chunk_len = static_cast<size_t>(chunk_length_secs * samp_freq);
		int32 check_period = static_cast<int32>(samp_freq * output_period);
		int32 check_count = check_period;

		int32 frame_offset = 0;

		bool eos = false;
		bool produce_time = false;

		OnlineNnet2FeaturePipeline feature_pipeline(*feature_info);
		SingleUtteranceNnet3Decoder decoder(*decoder_opts, *trans_model,
			*decodable_info,
			*decode_fst, &feature_pipeline);

		while (!eos) {

			decoder.InitDecoding(frame_offset);

			OnlineSilenceWeighting silence_weighting(
				*trans_model,
				feature_info->silence_weighting_config,
				decodable_opts->frame_subsampling_factor);

			std::vector<std::pair<int32, BaseFloat>> delta_weights;

			while (true) {

				eos = !ReadChunk(chunk_len);

				if (eos) {
					feature_pipeline.InputFinished();

					decoder.AdvanceDecoding();

					decoder.FinalizeDecoding();

					frame_offset += decoder.NumFramesDecoded();


					if (decoder.NumFramesDecoded() > 0) {
						CompactLattice lat;

						decoder.GetLattice(true, &lat);
						std::string msg = LatticeToString(lat, *word_syms);

						// get time-span from previous endpoint to end of audio,
						if (produce_time) {
							int32 t_beg = frame_offset - decoder.NumFramesDecoded();
							int32 t_end = frame_offset;
							msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " " + msg;
						}

						KALDI_VLOG(1) << "EndOfAudio, sending message: " << msg;
						WriteLn(msg);
					}
					else
						Write("\n");
					Disconnect();
					break;
				}

				Vector<BaseFloat> wave_part = GetChunk();
				feature_pipeline.AcceptWaveform(samp_freq, wave_part);
				samp_count += chunk_len;

				if (silence_weighting.Active() &&
					feature_pipeline.IvectorFeature() != NULL) {
					silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
					silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
						frame_offset * decodable_opts->frame_subsampling_factor,
						&delta_weights);
					feature_pipeline.UpdateFrameWeights(delta_weights);
				}

				decoder.AdvanceDecoding();

				if (samp_count > check_count) {
					if (decoder.NumFramesDecoded() > 0) {
						Lattice lat;
						decoder.GetBestPath(false, &lat);
						TopSort(&lat); // for LatticeStateTimes(),
						std::string msg = LatticeToString(lat, *word_syms);

						// get time-span after previous endpoint,
						if (produce_time) {
							int32 t_beg = frame_offset;
							int32 t_end = frame_offset + GetLatticeTimeSpan(lat);
							msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " " + msg;
						}

						KALDI_VLOG(1) << "Temporary transcript: " << msg;
						WriteLn(msg, "\r");
					}
					check_count += check_period;
				}

				if (decoder.EndpointDetected(*endpoint_opts)) {
					decoder.FinalizeDecoding();
					frame_offset += decoder.NumFramesDecoded();
					CompactLattice lat;
					decoder.GetLattice(true, &lat);
					std::string msg = LatticeToString(lat, *word_syms);

					// get time-span between endpoints,
					if (produce_time) {
						int32 t_beg = frame_offset - decoder.NumFramesDecoded();
						int32 t_end = frame_offset;
						msg = GetTimeString(t_beg, t_end, frame_shift * frame_subsampling) + " " + msg;
					}

					KALDI_VLOG(1) << "Endpoint, sending message: " << msg;
					WriteLn(msg);
					break; // while (true)
				}
			}
		}
	}

	bool DecodingThread::ReadChunk(size_t len) {

		if (buf_len_ != len) {
			buf_len_ = len;
			delete[] samp_buf_;
			samp_buf_ = new int16[len];
		}

		ssize_t ret;
		int poll_ret;
		size_t to_read = len;
		has_read_ = 0;

		while (to_read > 0) {
			poll_ret = poll(client_set_, 1, read_timeout_);
			if (poll_ret == 0) {
				KALDI_WARN << "Socket timeout! Disconnecting...";
				break;
			}
			if (poll_ret < 0) {
				KALDI_WARN << "Socket error! Disconnecting...";
				break;
			}
			ret = read(client_desc_, static_cast<void *>(samp_buf_ + has_read_), to_read * sizeof(int16));
			if (ret <= 0) {
				KALDI_WARN << "Stream over...";
				break;
			}
			to_read -= ret / sizeof(int16);
			has_read_ += ret / sizeof(int16);
		}

		return has_read_ > 0;
	}
	
	std::string DecodingThread::ReadKey(size_t len) {

		char key_buff[len];

		ssize_t ret;
		int poll_ret;
		size_t to_read = len;
		size_t bytes_read_ = 0;

		while (to_read > 0) {
			poll_ret = poll(client_set_, 1, read_timeout_);
			if (poll_ret == 0) {
				KALDI_WARN << "Key reading. Socket timeout! Disconnecting...";
				break;
			}
			if (poll_ret < 0) {
				KALDI_WARN << "Key reading. Socket error! Disconnecting...";
				break;
			}
			ret = read(client_desc_, static_cast<void *>(key_buff + bytes_read_), to_read * sizeof(char));
			if (ret <= 0) {
				KALDI_WARN << "Key reading. Stream over...";
				break;
			}
			to_read -= ret / sizeof(char);
			bytes_read_ += ret / sizeof(char);
		}

		std::string key(key_buff, len);
		return key;
	}

	Vector<BaseFloat> DecodingThread::GetChunk() {
		Vector<BaseFloat> buf;

		buf.Resize(static_cast<MatrixIndexT>(has_read_));

		for (int i = 0; i < has_read_; i++)
			buf(i) = static_cast<BaseFloat>(samp_buf_[i]);

		return buf;
	}

	bool DecodingThread::Write(const std::string &msg) {

		const char *p = msg.c_str();
		size_t to_write = msg.size();
		size_t wrote = 0;
		while (to_write > 0) {
			ssize_t ret = write(client_desc_, static_cast<const void *>(p + wrote), to_write);
			if (ret <= 0)
				return false;

			to_write -= ret;
			wrote += ret;
		}

		return true;
	}

	bool DecodingThread::WriteLn(const std::string &msg, const std::string &eol) {
		if (Write(msg))
			return Write(eol);
		else return false;
	}

	void DecodingThread::Disconnect() {
		if (samp_buf_ != NULL)
		{
			delete[] samp_buf_;
		}

		if (client_desc_ != -1) {
			close(client_desc_);
			client_desc_ = -1;
		}
	}

	DecodingThread::~DecodingThread()
	{

	}  // namespace kaldi
}