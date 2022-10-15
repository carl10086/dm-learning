## intro

- thread pool model
- Server Builder & Server
- each config for what ..
- quickstart for grpc

## generate code from grpc proto file

```protobuf
syntax = "proto3";

package com.ysz.arm.blog.client;
option java_package = "com.ysz.arm.blog.client";
option java_multiple_files = true;

service BlogService {
  rpc createBlogPost (CreateBlogPostRequest) returns (BlogPost) {}
}

message CreateBlogPostRequest {
  string title = 1;
  string content = 2;
}


message BlogPost {
  int32 id = 1;
  string title = 2;
  string content = 3;
  int64 createdAt = 4;
  int64 modifiedAT = 5;
}
```

- refer : [java grpc](https://grpc.io/docs/languages/java/basics/)
- why need java_package : because grpc `package` is not perfect for java pkgs ;
- grpc file is easy to understand .


we can use maven plugin , it's the recommend way it **production env .**

```xml
  <build>
    <extensions>
      <extension>
        <groupId>kr.motd.maven</groupId>
        <artifactId>os-maven-plugin</artifactId>
        <version>1.6.2</version>
      </extension>
    </extensions>
    <plugins>
      <plugin>
        <groupId>org.xolstice.maven.plugins</groupId>
        <artifactId>protobuf-maven-plugin</artifactId>
        <version>0.6.1</version>
        <configuration>
          <protocArtifact>com.google.protobuf:protoc:3.21.1:exe:${os.detected.classifier}</protocArtifact>
          <pluginId>grpc-java</pluginId>
          <pluginArtifact>io.grpc:protoc-gen-grpc-java:1.48.1:exe:${os.detected.classifier}</pluginArtifact>
        </configuration>
        <executions>
          <execution>
            <goals>
              <goal>compile</goal>
              <goal>compile-custom</goal>
            </goals>
          </execution>
        </executions>
      </plugin>
    </plugins>
  </build>
```

- in 2018. fuck .

## server & client 

- hello server
- hello client
- thread pool to execute server
- about dt-monitor plan


## monitor

**refer**

- it is removed now . fuck . [how to get threadlocal variables for async thrift service](https://github.com/line/armeria/issues/1067)
- removed in [2375](https://github.com/line/armeria/pull/2375)
- hooks back to Request Context in [2514](https://github.com/line/armeria/issues/2514)


