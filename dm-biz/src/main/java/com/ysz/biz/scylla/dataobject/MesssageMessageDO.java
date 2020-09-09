package com.ysz.biz.scylla.dataobject;

import com.datastax.driver.mapping.annotations.Table;
import java.util.Date;
import lombok.Data;

@Data
@Table(keyspace = "carl_test", name = "message_message")
public class MesssageMessageDO {

  private Integer id;
  private Date addDatetime;

  private Integer albumId;
  private Short category;

  private Integer flag;

  private Integer groupId;
  private Date lastRepliedDatetime;
  private Integer parentId;
  private Integer photoId;
  private Integer recipientId;
  private Integer senderId;
  private Integer sourceId;
  private Short status;
  private Integer threading;
  private Date updateAt;


}
