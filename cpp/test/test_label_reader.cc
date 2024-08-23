// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#include "graphar/label.h"
#include <chrono>

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>


// for chunkreader
#include <cstdlib>

#include "arrow/api.h"

#include "./util.h"
#include "graphar/api/arrow_reader.h"

#include <catch2/catch_test_macros.hpp>

// multi thread speedup
#include <thread>
#include <vector>
#include <future>
#include <condition_variable>

/// constants related to the test
// 1. cyber-security-ad-44-nodes.csv - too small
// row number: 954, label number: 7, test id: 0, 1, AND
// 2. graph-data-science-43-nodes.csv - poor
// row number: 2687, label number: 12, test id: 1, 6, AND
// 3. twitter-v2-43-nodes.csv
// row number: 43337, label number: 6, test id: 0, 1, AND
// 4. network-management-43-nodes.csv
// row number: 83847, label number: 17, test id: 4, 5, AND
// 5. fraud-detection-43-nodes.csv
// row number: 333022, label number: 13, test id: 3, 5, AND

// new datasets
// 1. legis-graph-43-nodes.csv, 11825, 8, {0, 1}, OR
// 2. recommendations-43-nodes.csv, 33880, 6, {3, 4}, AND - poor
// 3. bloom-43-nodes.csv, 32960, 18, {8, 9}, OR
// 4. pole-43-nodes.csv, 61534, 11, {1, 5}, OR
// 5. openstreetmap-43-nodes.csv, 71566, 10, {2, 5}, AND
// 6. icij-paradise-papers-43-nodes.csv, 163414, 5, {0, 4}, OR
// 7. citations-43-nodes.csv, 263902, 3, {0, 2}, OR
// 8. twitter-trolls-43-nodes.csv, 281177, 6, {0, 1}, AND
// 9. icij-offshoreleaks-44-nodes.csv, 1969309, 5, {1, 3}, OR
// 10. twitch-43-nodes.csv, 10516173, 5, {0, 1}, AND - poor

// ldbc datasets
// 1. place_0_0.csv, 1460, 4, {0, 2}, AND
// 2. organisation_0_0.csv, 7955, 3, {0, 1}, AND

// ogb datasets
// 1. ogbn-arxiv.csv, 169343, 40, {0, 1}, OR
// 2. ogbn-proteins.csv, 132534, 112, {0, 1}, AND
// 3. ogbn-mag.csv, 736389, 349, {0, 1}, OR
// 4. ogbn-products.csv, 2449029, 47, {0, 1}, OR
// 5. ogbn-papers100M.csv, 111059956, 172, {0, 1}, OR


const int TEST_ROUNDS = 1;                          // the number of test rounds
const int TOT_ROWS_NUM = 69165;                         // the number of total vertices
const int TOT_LABEL_NUM = 10;                         // the number of total labels
const int TESTED_LABEL_NUM = 2;                       // the number of tested labels
const int CHUNK_SIZE = 1000;
int tested_label_ids[TESTED_LABEL_NUM] = {2, 5};      // the ids of tested labels
const QUERY_TYPE fix_query_type = QUERY_TYPE::COUNT;  // the query type

std::mutex count_mutex; // 用于保护对total_counts的并发访问
std::mutex mtx;
std::condition_variable cv;
int active_threads = 0;
const int MAX_THREADS = 1; // 设置最大线程数为2

const char PARQUET_FILENAME_RLE[] =
    "parquet_graphar_label_RLE.parquet";  // the RLE filename
const char PARQUET_FILENAME_PLAIN[] =
    "parquet_graphar_label_plain.parquet";  // the PLAIN filename
const char PARQUET_FILENAME_STRING[] =
    "parquet_graphar_label_string.parquet";  // the STRING filename
const char PARQUET_FILENAME_STRING_DICT[] =
    "parquet_graphar_label_string_dict.parquet";  // the STRING filename

/// The label names
// const std::string label_names[TOT_LABEL_NUM] = {"label0", "label1", "label2", "label3",
//                                                 "label4", "label5", "label6",
//                                                 "label7"};
std::string label_names[MAX_LABEL_NUM];

/// Tested label ids
// const int tested_label_ids[TESTED_LABEL_NUM] = {0, 2, 3, 7};

/// global variables
int32_t length[TOT_LABEL_NUM];
int32_t repeated_nums[TOT_LABEL_NUM][MAX_DECODED_NUM];
bool repeated_values[TOT_LABEL_NUM][MAX_DECODED_NUM];
//bool** label_column_data;
bool label_column_data[TOT_ROWS_NUM][MAX_LABEL_NUM];

/// The user-defined function to check if the state is valid.
/// A default implementation is provided here, which checks if all labels are contained.
static inline bool IsValid(bool* state, int column_number) {
  for (int i = 0; i < column_number; ++i) {
    // AND case
    if (!state[i]) return false;
    // OR case
    // if (state[i]) return true;
  }
  // AND case
  return true;
  // OR case
//   return false;
}


/// Generate data of label columns for the parquet file (using bool datatype).
void generate_label_column_data_bool(const int num_rows, const int num_columns);

/// A hard-coded test for the function GetValidIntervals.
void hard_coded_test();

/// The test using string encoding/decoding for GraphAr labels.
void string_test(bool validate = false, bool enable_dictionary = false) {
  const char* filename;
  std::cout << "----------------------------------------\n";
  std::cout << "Running string test ";
  if (enable_dictionary) {
    std::cout << "with dictionary...\n";
    filename = PARQUET_FILENAME_STRING_DICT;
  } else {
    std::cout << "without dictionary...\n";
    filename = PARQUET_FILENAME_STRING;
  }

  // generate parquet file by ParquetWriter
  generate_parquet_file_string(filename, TOT_ROWS_NUM, TOT_LABEL_NUM, label_column_data,
                               label_names, false, enable_dictionary);
  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    std::chrono::duration<double> duration = end - start;

    // 输出执行时间（单位：秒）
    std::cout << "代码执行时间: " << duration.count() << " 秒" << std::endl;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::ADAPTIVE || fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          label_names, IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (size_t i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}



void process_chunk(int chunk_idx, const std::string& base_filename,
                  int TOT_ROWS_NUM, int TOT_LABEL_NUM, int TESTED_LABEL_NUM,
                  const int* tested_label_ids,  uint64_t* bitmap,
                  std::vector<int>& total_counts, const std::function<bool(bool*, int)>& IsValid) {
    std::string new_filename = base_filename + "/chunk" + std::to_string(chunk_idx);
    std::vector<int> indices;
    int count = read_parquet_file_and_get_valid_indices(
        new_filename.c_str(), TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
        IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
    std::lock_guard<std::mutex> lock(count_mutex);
    total_counts.push_back(count);
    std::cout << "Chunk " << chunk_idx << ": Processed " << count << " valid indices." << std::endl;
     {
        std::lock_guard<std::mutex> lock(mtx);
        --active_threads;
        cv.notify_one(); // 通知等待的线程
    }
}

/// The test using bool plain encoding/decoding for GraphAr labels.

void bool_plain_test(bool validate = false,
                     const char* filename = PARQUET_FILENAME_PLAIN) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running baseline bool test ";
//   if (filename == PARQUET_FILENAME_PLAIN) {
//     std::cout << "(plain)..." << std::endl;
//     // generate parquet file by ParquetWriter
//     generate_parquet_file_bool_plain(PARQUET_FILENAME_PLAIN, TOT_ROWS_NUM, TOT_LABEL_NUM,
//                                      label_column_data, label_names, false);
//   } else {
//     std::cout << "(RLE)..." << std::endl;
//     // generate parquet file by ParquetWriter
//     generate_parquet_file_bool_RLE(PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM,
//                                    label_column_data, label_names, false);
//   }

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_ROUNDS; i++) {
    //   int total_count = 0;
    //   for (int chunk_idx = 0; chunk_idx * CHUNK_SIZE < TOT_ROWS_NUM; ++chunk_idx) {
    //     std::string new_filename = std::string(filename) + "/chunk" + std::to_string(chunk_idx);
    //     // std::string new_filename = std::string(filename) +  "parquet_graphar_label_RLE.parquet";
    //     // 注意：这里需要确保IsValidFunc类型的IsValid已正确定义并可用
    //     std::vector<int> indices;
    //     int count = read_parquet_file_and_get_valid_indices(
    //         new_filename.c_str(), TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
    //         IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
    //     total_count += count;
        
    //     std::cout << "Chunk " << chunk_idx << ": Processed " << count << " valid indices." << std::endl;
    //     // 这里可以进一步处理indices和count，比如累加计数或处理有效索引
    // }
    // std::cout << "Total valid count: " << total_count << std::endl;
    std::vector<std::thread> threads;
    std::vector<int> total_counts;
    

    for (int chunk_idx = 0; chunk_idx * CHUNK_SIZE < TOT_ROWS_NUM; ++chunk_idx) {
        std::unique_lock<std::mutex> lock(mtx);
          // 等待直到有线程完成
        cv.wait(lock, []{ return active_threads < MAX_THREADS; });
        ++active_threads;
        threads.emplace_back(process_chunk, chunk_idx, filename,
                            TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
                            tested_label_ids, bitmap, std::ref(total_counts), std::ref(IsValid));
        // process_chunk(chunk_idx, filename,
        //                     TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
        //                     tested_label_ids, bitmap, total_counts, IsValid);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 计算总处理计数
    int total_count = 0;
    for (int count : total_counts) {
        total_count += count;
    }
    std::cout << "Total processed valid indices: " << total_count << std::endl;

    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    std::chrono::duration<double> duration = end - start;
    std::cout << "代码执行时间: " << duration.count() << " 秒" << std::endl;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
  }
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::ADAPTIVE || fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      for (int chunk_idx = 0; chunk_idx * CHUNK_SIZE < TOT_ROWS_NUM; ++chunk_idx) {
        std::string new_filename = std::string(filename) + "/chunk" + std::to_string(chunk_idx);
        
        // 注意：这里需要确保IsValidFunc类型的IsValid已正确定义并可用
        std::vector<int> indices;
        int count = read_parquet_file_and_get_valid_indices(
            new_filename.c_str(), TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
            IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
        
        std::cout << "Chunk " << chunk_idx << ": Processed " << count << " valid indices." << std::endl;
        // 这里可以进一步处理indices和count，比如累加计数或处理有效索引
    }
      // count = read_parquet_file_and_get_valid_indices(
      //     filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
      //     IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
    }
   
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      for (int chunk_idx = 0; chunk_idx * CHUNK_SIZE < TOT_ROWS_NUM; ++chunk_idx) {
        std::string new_filename = std::string(filename) + "/chunk" + std::to_string(chunk_idx);
        
        // 注意：这里需要确保IsValidFunc类型的IsValid已正确定义并可用
        std::vector<int> indices;
        int count = read_parquet_file_and_get_valid_indices(
            new_filename.c_str(), TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
            IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
        
        std::cout << "Chunk " << chunk_idx << ": Processed " << count << " valid indices." << std::endl;
        // 这里可以进一步处理indices和count，比如累加计数或处理有效索引
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;
  }

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (size_t i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

/// The test using bool plain encoding/decoding for GraphAr labels with chunk reader.

void bool_plain_test_chunk_reader(bool validate = false,
                     const char* filename = PARQUET_FILENAME_PLAIN) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running baseline bool test with chunk reader";
//   if (filename == PARQUET_FILENAME_PLAIN) {
//     std::cout << "(plain)..." << std::endl;
//     // generate parquet file by ParquetWriter
//     generate_parquet_file_bool_plain(PARQUET_FILENAME_PLAIN, TOT_ROWS_NUM, TOT_LABEL_NUM,
//                                      label_column_data, label_names, false);
//   } else {
//     std::cout << "(RLE)..." << std::endl;
//     // generate parquet file by ParquetWriter
//     generate_parquet_file_bool_RLE(PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM,
//                                    label_column_data, label_names, false);
//   }

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    auto start = std::chrono::high_resolution_clock::now();
    std::string test_data_dir = "/workspaces/incubator-graphar/cpp/build/testing";
    std::string prefix = test_data_dir + "/openstreet/";
    std::string vertex_info_path = prefix + "parquet/osm_node.vertex.yml";
    std::cout << "Vertex info path: " << vertex_info_path << std::endl;
    auto fs = graphar::FileSystemFromUriOrPath(prefix).value();
    auto yaml_content =
        fs->ReadFileToValue<std::string>(vertex_info_path).value();
    std::cout << yaml_content << std::endl;
    auto maybe_vertex_info = graphar::VertexInfo::Load(yaml_content);
    std::cout << "maybe_vertex_info: " << maybe_vertex_info.status().message() << std::endl;
    std::cout << maybe_vertex_info.status().ok() << std::endl;
    // REQUIRE(maybe_vertex_info.status().ok());
    auto vertex_info = maybe_vertex_info.value();
    std::vector<std::string> label_names = {
        "OSM",
        "Bounds",
        "OSMNode",
        "Intersection",
        "OSMTags",
        "PointOfInterest",
        "Routable",
        "OSMWay",
        "OSMWayNode",
        "OSMRelation"
    };
    auto maybe_reader =
          graphar::VertexPropertyArrowChunkReader::Make(vertex_info, label_names, prefix, {});
    // for (int i = 0; i < TEST_ROUNDS; i++) {
    //   // count = read_parquet_file_and_get_valid_indices(
    //   //     filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
    //   //     IsValid, &indices, bitmap, QUERY_TYPE::COUNT);
    //   REQUIRE(maybe_reader.status().ok());
    auto reader = maybe_reader.value();
    //   auto result = reader->GetChunk();

    // }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    std::chrono::duration<double> duration = end - start;
    std::cout << "代码执行时间: " << duration.count() << " 秒" << std::endl;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::ADAPTIVE || fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          IsValid, &indices, bitmap, QUERY_TYPE::INDEX);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_indices(
          filename, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM, tested_label_ids,
          IsValid, &indices, bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (size_t i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

/// The test using bool RLE encoding/decoding for GraphAr labels.
void RLE_test(bool validate = false) {
  std::cout << "----------------------------------------\n";
  std::cout << "Running optimized RLE test..." << std::endl;

  // allocate memory
  std::vector<int> indices;
  uint64_t* bitmap = new uint64_t[TOT_ROWS_NUM / 64 + 1];
  memset(bitmap, 0, sizeof(uint64_t) * (TOT_ROWS_NUM / 64 + 1));
  int count;

  // test getting count
  if (fix_query_type == QUERY_TYPE::COUNT) {
    auto run_start = clock();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::COUNT);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    auto end = std::chrono::high_resolution_clock::now();
    // 计算时间差
    std::chrono::duration<double> duration = end - start;
    std::cout << "代码执行时间: " << duration.count() << " 秒" << std::endl;
    std::cout << "[Performance] The run time for the test (COUNT) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test adaptive mode
  if (fix_query_type == QUERY_TYPE::ADAPTIVE) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::ADAPTIVE);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (ADAPTIVE) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting indices
  if (fix_query_type == QUERY_TYPE::INDEX) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::INDEX);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (INDEX) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  // test getting bitmap
  if (fix_query_type == QUERY_TYPE::BITMAP) {
    auto run_start = clock();
    for (int i = 0; i < TEST_ROUNDS; i++) {
      count = read_parquet_file_and_get_valid_intervals(
          PARQUET_FILENAME_RLE, TOT_ROWS_NUM, TOT_LABEL_NUM, TESTED_LABEL_NUM,
          tested_label_ids, repeated_nums, repeated_values, length, IsValid, &indices,
          bitmap, QUERY_TYPE::BITMAP);
    }
    auto run_time = 1000.0 * (clock() - run_start) / CLOCKS_PER_SEC;
    std::cout << "[Performance] The run time for the test (BITMAP) is: " << run_time
              << " ms.\n"
              << std::endl;
  }

  std::cout << "The valid count is: " << count << std::endl;

  if (validate) {
    // print the results
    std::cout << "The valid indices are:" << std::endl;
    for (size_t i = 0; i < indices.size(); i++) {
      std::cout << indices[i] << std::endl;
      if (!GetBit(bitmap, indices[i])) {
        std::cout << "[Error] The index " << indices[i] << " is not valid." << std::endl;
      }
    }
  }

  delete[] bitmap;
  std::cout << "----------------------------------------\n";
}

// read csv and generate label column data
void read_csv_file_and_generate_label_column_data_bool(const int num_rows,
                                                       const int num_columns);

int main(int argc, char** argv) {
  unsigned num_cores = std::thread::hardware_concurrency(); // 获取系统核心数
  std::cout << "System has " << num_cores << " cores.\n";

  // 根据核心数创建线程，这里假设我们不希望超过核心数，可以根据实际情况调整
  const int THREAD_COUNT = std::min(num_cores, 4u); // 举例，最多创建4个线程
  //label_column_data = new bool*[TOT_ROWS_NUM];
  //for (int i = 0; i < TOT_ROWS_NUM; i++) {
  //  label_column_data[i] = new bool[TOT_LABEL_NUM];
  //}
  // hard-coded test
  // hard_coded_test();

  // generate label column data
  // generate_label_column_data_bool(TOT_ROWS_NUM, TOT_LABEL_NUM);

  // read csv and generate label column data
//   read_csv_file_and_generate_label_column_data_bool(TOT_ROWS_NUM, TOT_LABEL_NUM);

  // string test: disable dictionary
//   string_test();

  // string test: enable dictionary
  // string_test(false, true);

  // bool plain test
//   bool_plain_test();

  // bool test use RLE but not optimized
  bool_plain_test(false, "/workspaces/incubator-graphar/cpp/build/testing/openstreet/parquet/vertex/osm_node/labels/");

  // bool test use RLE but not optimized and read by chunk reader
  // bool_plain_test_chunk_reader(false, "/workspaces/incubator-graphar/cpp/build/testing/openstreet/parquet/vertex/osm_node/labels/");
  // bool test use RLE and optimized
//   RLE_test();

  return 0;
}

void read_csv_file_and_generate_label_column_data_bool(const int num_rows,
                                                       const int num_columns) {
  // the first line is the header
  for (int k = 0; k < num_columns; k++) {
    std::cin >> label_names[k];
  }
  // read the data
  int cnt[TOT_LABEL_NUM] = {0};
  for (int index = 0; index < num_rows; index++) {
    for (int k = 0; k < num_columns; k++) {
      std::cin >> label_column_data[index][k];
      if (label_column_data[index][k] == 1) {
        cnt[k]++;
      }
    }
  }
}

/* void generate_label_column_data_bool(const int num_rows, const int num_columns) {
  for (int index = 0; index < num_rows; index++) {
    for (int k = 0; k < num_columns; k++) {
      bool value;
      if (index < 50) {
        value = true;
      } else if (k == tested_label_ids[0]) {
        value = true;
      } else if (k == tested_label_ids[1]) {
        value = (index % 3333 < 999) ? true : false;
      } else if (k == tested_label_ids[2]) {
        value = ((index / 10) % 10 == 0) ? true : false;
      } else if (k == tested_label_ids[3]) {
        value = ((index / 100) % 2 == 1) ? true : false;
      } else {
        value = true;
      }
      label_column_data[index][k] = value;
    }
  }
} */

void hard_coded_test() {
  std::cout << "----------------------------------------\n";
  std::cout << "Running hard-coded test..." << std::endl;

  // set length, repeated_nums, repeated_values
  // num_rows = 13, num_columns = 3
  length[0] = 3;
  length[1] = 4;
  length[2] = 6;
  repeated_nums[0][0] = 6;
  repeated_values[0][0] = 1;
  repeated_nums[0][1] = 6;
  repeated_values[0][1] = 1;
  repeated_nums[0][2] = 1;
  repeated_values[0][2] = 1;
  repeated_nums[1][0] = 4;
  repeated_values[1][0] = 0;
  repeated_nums[1][1] = 5;
  repeated_values[1][1] = 1;
  repeated_nums[1][2] = 3;
  repeated_values[1][2] = 0;
  repeated_nums[1][3] = 1;
  repeated_values[1][3] = 1;
  repeated_nums[2][0] = 2;
  repeated_values[2][0] = 0;
  repeated_nums[2][1] = 2;
  repeated_values[2][1] = 1;
  repeated_nums[2][2] = 1;
  repeated_values[2][2] = 0;
  repeated_nums[2][3] = 3;
  repeated_values[2][3] = 1;
  repeated_nums[2][4] = 4;
  repeated_values[2][4] = 0;
  repeated_nums[2][5] = 1;
  repeated_values[2][5] = 1;

  auto count = GetValidIntervals(3, 13, repeated_nums, repeated_values, length, IsValid);

  std::cout << "The valid count is: " << count << std::endl;
  std::cout << "----------------------------------------\n";
}

