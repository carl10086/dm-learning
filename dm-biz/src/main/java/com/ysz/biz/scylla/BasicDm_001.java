package com.ysz.biz.scylla;

import com.datastax.driver.core.Cluster;
import com.datastax.driver.core.Session;
import com.datastax.driver.core.policies.RoundRobinPolicy;
import com.datastax.driver.mapping.DefaultNamingStrategy;
import com.datastax.driver.mapping.DefaultPropertyMapper;
import com.datastax.driver.mapping.Mapper;
import com.datastax.driver.mapping.MappingConfiguration;
import com.datastax.driver.mapping.MappingManager;
import com.datastax.driver.mapping.NamingConventions;
import com.datastax.driver.mapping.PropertyMapper;
import com.ysz.biz.scylla.dataobject.MesssageMessageDO;
import java.util.Date;

public class BasicDm_001 {


  public static void main(String[] args) {
    try (Cluster cluster = Cluster.builder().addContactPoints("10.1.1.36", "10.1.1.47", "10.1.1.96")
        .withLoadBalancingPolicy(new RoundRobinPolicy())
        .build(); Session session = cluster.connect()
    ) {
      /*java 是驼峰, 数据库是 下划线*/
      PropertyMapper propertyMapper = new DefaultPropertyMapper()
          .setNamingStrategy(new DefaultNamingStrategy(
              NamingConventions.LOWER_CAMEL_CASE,
              NamingConventions.LOWER_SNAKE_CASE));
      MappingConfiguration configuration = MappingConfiguration.builder()
          .withPropertyMapper(propertyMapper).build();
      MappingManager mappingManager = new MappingManager(session, configuration);
      Mapper<MesssageMessageDO> mapper = mappingManager.mapper(MesssageMessageDO.class);
      mapper.save(mockMesssageMessageDO());
    }
  }


  private static MesssageMessageDO mockMesssageMessageDO() {
    MesssageMessageDO messsageMessageDO = new MesssageMessageDO();
    messsageMessageDO.setId(1);
    messsageMessageDO.setAddDatetime(new Date());
    messsageMessageDO.setAlbumId(1);
    short category = 1;
    messsageMessageDO.setCategory(category);
    messsageMessageDO.setFlag(1);
    messsageMessageDO.setGroupId(1);
    messsageMessageDO.setLastRepliedDatetime(new Date());
    messsageMessageDO.setParentId(null);
    messsageMessageDO.setPhotoId(1);
    messsageMessageDO.setSenderId(1000);
    messsageMessageDO.setSourceId(10001);
    short status = 0;
    messsageMessageDO.setStatus(status);
    messsageMessageDO.setUpdateAt(new Date());
    return messsageMessageDO;
  }

}
