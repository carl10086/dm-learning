package com.ysz.biz.mysql.binlog;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class MBlogLikeEvent {

  private Long senderId;

  /**
   * true 是 like
   * false 是 unlike
   */
  private boolean like;

  private Long blogId;

  private String timeStr;


  @Override
  public String toString() {
    return senderId + "," + (like ? "like" : "unlike") + "," + blogId + "," + timeStr;
  }
}
