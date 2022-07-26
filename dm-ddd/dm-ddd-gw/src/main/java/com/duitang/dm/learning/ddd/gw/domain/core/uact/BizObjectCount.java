package com.duitang.dm.learning.ddd.gw.domain.core.uact;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

/**
 * 一个实体 在 用户行为下的计数 .
 */
@ToString
@Getter
public class BizObjectCount {

  private Long id;

  private Integer favCnt;

  private Integer likeCnt;

  private Integer commentCnt;

  private Integer commentUserCnt;

  private Instant lastFavAt;

  private Instant lastCommentAt;

  private Instant lastLikeAt;

  public BizObjectCount setId(Long id) {
    this.id = id;
    return this;
  }

  public BizObjectCount setFavCnt(Integer favCnt) {
    this.favCnt = favCnt;
    return this;
  }

  public BizObjectCount setLikeCnt(Integer likeCnt) {
    this.likeCnt = likeCnt;
    return this;
  }

  public BizObjectCount setCommentCnt(Integer commentCnt) {
    this.commentCnt = commentCnt;
    return this;
  }

  public BizObjectCount setCommentUserCnt(Integer commentUserCnt) {
    this.commentUserCnt = commentUserCnt;
    return this;
  }

  public BizObjectCount setLastFavAt(Instant lastFavAt) {
    this.lastFavAt = lastFavAt;
    return this;
  }

  public BizObjectCount setLastCommentAt(Instant lastCommentAt) {
    this.lastCommentAt = lastCommentAt;
    return this;
  }

  public BizObjectCount setLastLikeAt(Instant lastLikeAt) {
    this.lastLikeAt = lastLikeAt;
    return this;
  }
}
