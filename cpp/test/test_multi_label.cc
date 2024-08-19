/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <time.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "arrow/api.h"
#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/stl.h"
#include "arrow/util/uri.h"
#include "parquet/arrow/writer.h"

#include "./util.h"
#include "graphar/api/high_level_writer.h"

#include <catch2/catch_test_macros.hpp>
namespace graphar {
TEST_CASE_METHOD(GlobalFixture, "test_vertices_builder") {
  std::cout << "Test vertex builder" << std::endl;

  // construct graph information from file
  std::string path =
      test_data_dir + "/openstreet/parquet/" + "openstreet.graph.yml";
  auto graph_info = graphar::GraphInfo::Load(path).value();
  auto vertex_info = graph_info->GetVertexInfo("osm_node");
  std::vector<std::string> labels = graph_info->GetLabels();

  std::unordered_map<std::string, size_t> code;

  // 初始化哈希表code
  for (size_t i = 0; i < labels.size(); ++i) {
      code[labels[i]] = i;
  }

  std::vector<std::vector<bool>> label_column_data;
  std::string message = graph_info->Dump().value();
  std::cout << message << std::endl;
  
  // construct vertex builder
  // std::string vertex_meta_file =
  //     test_data_dir + "/openstreet/parquet/" + "osm_node.vertex.yml";
  // auto vertex_meta = Yaml::LoadFile(vertex_meta_file).value();
  // auto vertex_info = VertexInfo::Load(vertex_meta).value();
  IdType start_index = 0;
  auto maybe_builder =
      builder::VerticesBuilder::Make(vertex_info, "/tmp/", start_index);
  REQUIRE(!maybe_builder.has_error());
  auto builder = maybe_builder.value();

  // std::string message = vertex_info->Dump().value();
  // std::cout << message << std::endl;
  // std::cout << "Has id: " << vertex_info->HasProperty("id") << std::endl;
  // std::cout << "Has lon: " << vertex_info->HasProperty("lon") << std::endl;
  // std::cout << "Has lat: " << vertex_info->HasProperty("lat") << std::endl;

  // get & set validate level
  REQUIRE(builder->GetValidateLevel() == ValidateLevel::no_validate);
  builder->SetValidateLevel(ValidateLevel::strong_validate);
  REQUIRE(builder->GetValidateLevel() == ValidateLevel::strong_validate);


  // clear vertices
  builder->Clear();
  REQUIRE(builder->GetNum() == 0);

  // add vertices
  std::ifstream fp(test_data_dir + "/openstreet/openstreet.csv");
  std::string line;
  getline(fp, line);
  int m = 4;
  std::vector<std::string> names;
  // std::istringstream readstr(line);
  // for (int i = 0; i < m; i++) {
  //   std::string name;
  //   getline(readstr, name, ',');
  //   names.push_back(name);
  // }
  names.push_back("id");
  names.push_back("lon");
  names.push_back("lat");

  int lines = 0;
  while (getline(fp, line)) {
    lines++;
    std::string val;
    std::istringstream readstr(line);
    builder::Vertex v;
    for (int i = 0; i < m; i++) {
      getline(readstr, val, ',');
      if (i == 0) {
        int64_t x = 0;
        for (size_t j = 0; j < val.length(); j++)
          x = x * 10 + val[j] - '0';
        v.AddProperty(names[i], x);
      } else if (i == 1) { //labels
        std::istringstream labelStream(val);
        std::string singleLabel;
        std::vector<bool> tmp_labels(labels.size(), false);
        while (std::getline(labelStream, singleLabel, '|')) {
            tmp_labels[code[singleLabel]] = true;
        }
        label_column_data.push_back(tmp_labels);
      } else {
        v.AddProperty(names[i-1], val);
      }
    }
    REQUIRE(builder->AddVertex(v).ok());
  }
  std::ofstream output_file("code_labels.csv");
  if(output_file.is_open()){
      // 先输出列标题
      for(size_t j = 0; j < labels.size(); ++j) {
          output_file << labels[j];
          if(j != labels.size() - 1) output_file << " ";
      }
      output_file << "\n"; // 换行

      std::ostringstream oss; // 使用字符串流累积其他输出
      for(size_t i = 0; i < label_column_data.size(); i++){
          for(size_t j = 0; j < code.size(); j++){
              oss << label_column_data[i][j];
              if(j != code.size() - 1) oss << " ";
          }
          oss << "\n"; // 换行
      }
      output_file << oss.str(); // 一次性写入所有累积的数据
      output_file.close();
  }else{
      std::cout << "Unable to open file" << std::endl;
  }


  // check the number of vertices in builder
  REQUIRE(builder->GetNum() == lines);

  // dump to files
  REQUIRE(builder->Dump().ok());
  

  // can not add new vertices after dumping
  // REQUIRE(builder->AddVertex(v).IsInvalid());

  // check the number of vertices dumped
  auto fs = arrow::fs::FileSystemFromUriOrPath(test_data_dir).ValueOrDie();
  auto input =
      fs->OpenInputStream("/tmp/vertex/osm_node/vertex_count").ValueOrDie();
  auto num = input->Read(sizeof(IdType)).ValueOrDie();
  const IdType* ptr = reinterpret_cast<const IdType*>(num->data());
  REQUIRE((*ptr) == start_index + builder->GetNum());
}

TEST_CASE_METHOD(GlobalFixture, "test_edges_builder") {
  std::cout << "Test edge builder" << std::endl;
  // construct edge builder
  std::string edge_meta_file =
      test_data_dir + "/openstreet/parquet/" + "osm_node_next.edge.yml";
  auto edge_meta = Yaml::LoadFile(edge_meta_file).value();
  auto edge_info = EdgeInfo::Load(edge_meta).value();
  auto vertices_num = 69165;
  auto maybe_builder = builder::EdgesBuilder::Make(
      edge_info, "/tmp/", AdjListType::ordered_by_dest, vertices_num);
  REQUIRE(!maybe_builder.has_error());
  auto builder = maybe_builder.value();

  // get & set validate level
  REQUIRE(builder->GetValidateLevel() == ValidateLevel::no_validate);
  builder->SetValidateLevel(ValidateLevel::strong_validate);
  REQUIRE(builder->GetValidateLevel() == ValidateLevel::strong_validate);

  // // check different validate levels
  // builder::Edge e(0, 1);
  // e.AddProperty("creationDate", 2020);
  // REQUIRE(builder->AddEdge(e, ValidateLevel::no_validate).ok());
  // REQUIRE(builder->AddEdge(e, ValidateLevel::weak_validate).ok());
  // REQUIRE(builder->AddEdge(e, ValidateLevel::strong_validate).IsTypeError());
  // e.AddProperty("invalid_name", "invalid_value");
  // REQUIRE(builder->AddEdge(e).IsKeyError());

  // // clear edges
  // builder->Clear();
  // REQUIRE(builder->GetNum() == 0);

  // add edges
  std::ifstream fp(test_data_dir + "/openstreet/openstreet_next.csv");
  std::string line;
  getline(fp, line);
  std::vector<std::string> names;
  std::istringstream readstr(line);
  std::map<std::string, int64_t> mapping;
  int64_t cnt = 0, lines = 0;

  while (getline(fp, line)) {
    lines++;
    std::string val;
    std::istringstream readstr(line);
    int64_t s = 0, d = 0;
    for (int i = 0; i < 3; i++) {
      getline(readstr, val, ',');
      if (i == 0) {
        if (mapping.find(val) == mapping.end())
          mapping[val] = cnt++;
        s = mapping[val];
      } else if (i == 1) {
        if (mapping.find(val) == mapping.end())
          mapping[val] = cnt++;
        d = mapping[val];
      } else {
        builder::Edge e(s, d);
        e.AddProperty("distance", val);
        REQUIRE(builder->AddEdge(e).ok());
      }
    }
  }

  // check the number of edges in builder
  REQUIRE(builder->GetNum() == lines);

  // dump to files
  REQUIRE(builder->Dump().ok());

  // can not add new edges after dumping
  // REQUIRE(builder->AddEdge(e).IsInvalid());

  // check the number of vertices dumped
  auto fs = arrow::fs::FileSystemFromUriOrPath(test_data_dir).ValueOrDie();
  auto input =
      fs->OpenInputStream(
            "/tmp/edge/osm_node_next/ordered_by_dest/vertex_count")
          .ValueOrDie();
  auto num = input->Read(sizeof(IdType)).ValueOrDie();
  const IdType* ptr = reinterpret_cast<const IdType*>(num->data());
  REQUIRE((*ptr) == vertices_num);
}
}  // namespace graphar
