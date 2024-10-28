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

package org.apache.graphar.info.saver;

import java.io.IOException;
import org.apache.graphar.info.EdgeInfo;
import org.apache.graphar.info.GraphInfo;
import org.apache.graphar.info.VertexInfo;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class LocalYamlGraphSaver implements GraphSaver {
    private static final FileSystem fileSystem;

    static {
        try {
            fileSystem = FileSystem.get(new Configuration());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void save(String path, GraphInfo graphInfo) throws IOException {
        try (FSDataOutputStream outputStream =
                fileSystem.create(new Path(path + "/" + graphInfo.getName() + ".graph.yaml"))) {
            outputStream.writeBytes(graphInfo.dump());
            for (VertexInfo vertexInfo : graphInfo.getVertexInfos()) {
                saveVertex(path, vertexInfo);
            }
            for (EdgeInfo edgeInfo : graphInfo.getEdgeInfos()) {
                saveEdge(path, edgeInfo);
            }
        }
    }

    private void saveVertex(String path, VertexInfo vertexInfo) throws IOException {
        try (FSDataOutputStream outputStream =
                fileSystem.create(new Path(path + "/" + vertexInfo.getType() + ".vertex.yaml"))) {
            outputStream.writeBytes(vertexInfo.dump());
        }
    }

    private void saveEdge(String path, EdgeInfo edgeInfo) throws IOException {
        try (FSDataOutputStream outputStream =
                fileSystem.create(new Path(path + "/" + edgeInfo.getConcat() + ".edge.yaml"))) {
            outputStream.writeBytes(edgeInfo.dump());
        }
    }
}