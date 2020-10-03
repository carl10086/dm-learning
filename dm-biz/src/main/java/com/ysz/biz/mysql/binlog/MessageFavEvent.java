package com.ysz.biz.mysql.binlog;

import java.util.Date;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MessageFavEvent {

  private Long blogId;
  private String occurAt;
  private String type;

  private Long nextPosition;
}
