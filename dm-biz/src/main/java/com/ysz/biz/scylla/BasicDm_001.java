package com.ysz.biz.scylla;

import com.datastax.driver.core.Cluster;
import com.datastax.driver.core.ResultSet;
import com.datastax.driver.core.Row;
import com.datastax.driver.core.Session;
import com.datastax.driver.core.policies.RoundRobinPolicy;
import com.datastax.driver.core.querybuilder.Clause;
import com.datastax.driver.core.querybuilder.QueryBuilder;
import com.datastax.driver.core.querybuilder.Select.Where;
import com.datastax.driver.mapping.DefaultNamingStrategy;
import com.datastax.driver.mapping.DefaultPropertyMapper;
import com.datastax.driver.mapping.Mapper;
import com.datastax.driver.mapping.MappingConfiguration;
import com.datastax.driver.mapping.MappingManager;
import com.datastax.driver.mapping.NamingConventions;
import com.datastax.driver.mapping.PropertyMapper;
import com.google.common.collect.Lists;
import com.ysz.biz.scylla.dataobject.MesssageMessageDO;
import java.util.Date;

public class BasicDm_001 {


  public static void main(String[] args) {
    try (Cluster cluster = Cluster.builder().addContactPoints("10.1.4.36")
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
      Clause id = QueryBuilder.in("id",
          Lists.newArrayList(
              1
          )
      );

      Mapper<MesssageMessageDO> mapper = mappingManager.mapper(MesssageMessageDO.class);
      mapper.save(mockMesssageMessageDO());

      Where where = QueryBuilder.select().from("carl_test", "message_message").where(id);
      long start = System.currentTimeMillis();
      int count = 100;

      for (int i = 0; i < count; i++) {
        ResultSet execute = session.execute(where);
        for (Row row : execute) {
          assert row != null;
        }
      }
      long end = System.currentTimeMillis();
      System.err.println((end - start) / count);

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
