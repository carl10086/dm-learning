package com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.impl;

import com.ysz.dm.duitang.srv.support.blog.domain.BlogForward;
import com.ysz.dm.duitang.srv.support.blog.domain.BlogForwardRepo;
import com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.converter.BlogConverter;
import com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.dao.BlogForwardDAO;
import com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.dao.BlogMetaDAO;
import com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.dataobject.BlogForwardDO;
import com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.dataobject.BlogMetaDO;
import com.ysz.dm.duitang.srv.support.blog.port.adapater.persist.mapper.BlogMapper;

public class BlogForwardRepoImpl implements BlogForwardRepo {

  private BlogForwardDAO blogForwardDAO;

  private BlogMetaDAO blogMetaDAO;

  private BlogConverter blogConverter;

  private BlogMapper blogMapper;

  @Override
  public void save(final BlogForward blogForward) {
    BlogForwardDO blogForwardDO = blogMapper.toBlogForwardDO(blogForward);
    blogForwardDAO.insertOne(blogForwardDO);
  }

  @Override
  public BlogForward findOne(final Long forwardId) {
    BlogForwardDO blogForwardDO = blogForwardDAO.selectOneById(forwardId);
    BlogMetaDO blogMetaDO = blogMetaDAO.selectOneById(blogForwardDO.getRootId());
    return blogConverter.toBlogForward(blogForwardDO, blogMetaDO);
  }
}
