package com.ysz.dm.duitang.srv.support.blog.domain.adapter;

import com.ysz.dm.duitang.srv.infra.anno.DddValueObj;

@DddValueObj
public class BlogCount {

  private final int likeCnt;
  private final int favCnt;
  private final int commentCnt;

  public BlogCount(final int likeCnt, final int favCnt, final int commentCnt) {
    this.likeCnt = likeCnt;
    this.favCnt = favCnt;
    this.commentCnt = commentCnt;
  }
}
