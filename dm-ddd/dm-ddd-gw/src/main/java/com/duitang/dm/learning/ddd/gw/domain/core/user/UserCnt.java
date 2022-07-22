package com.duitang.dm.learning.ddd.gw.domain.core.user;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class UserCnt {

  private Integer pubAtlasCnt;
  private Integer commentCnt;


  private Instant lastPubAtlasAt;
  private Instant lastCommentAt;


  public UserCnt setPubAtlasCnt(Integer pubAtlasCnt) {
    this.pubAtlasCnt = pubAtlasCnt;
    return this;
  }

  public UserCnt setCommentCnt(Integer commentCnt) {
    this.commentCnt = commentCnt;
    return this;
  }

  public UserCnt setLastPubAtlasAt(Instant lastPubAtlasAt) {
    this.lastPubAtlasAt = lastPubAtlasAt;
    return this;
  }

  public UserCnt setLastCommentAt(Instant lastCommentAt) {
    this.lastCommentAt = lastCommentAt;
    return this;
  }
}
