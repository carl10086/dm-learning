package com.ysz.codemaker.mybatis.core;

import com.ysz.codemaker.toos.common.JavaClassId;
import com.ysz.codemaker.toos.mysql.core.MysqlCfg;
import java.io.File;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class Cfg {

  private MysqlCfg mysql;
  private String database;
  private String tableName;


  /**
   * 表字段 和  JAVA 字段名称的映射关系
   */
  private DefaultMappingStrategy mappingStrategy = DefaultMappingStrategy.LOWER_UNDERSCORE_2_LOWER_CAMEL;

  /**
   * 当指定了 versionColName 的时候. 后续都会自动生成一个方法, 叫做 update by version
   */
  private String versionColName;

  private JavaClassId dataObjectClass;

  private String mapperXmlMustache = classpathFile("tpl/mybatis/mapper_xml.mustache");


  public static String classpathFile(String path) {
    return Cfg.class.getClassLoader().getResource(path).getFile();
  }


  public Cfg setMysql(MysqlCfg mysql) {
    this.mysql = mysql;
    return this;
  }

  public Cfg setDatabase(String database) {
    this.database = database;
    return this;
  }

  public Cfg setTableName(String tableName) {
    this.tableName = tableName;
    return this;
  }

  public Cfg setMappingStrategy(DefaultMappingStrategy mappingStrategy) {
    this.mappingStrategy = mappingStrategy;
    return this;
  }

  public Cfg setVersionColName(String versionColName) {
    this.versionColName = versionColName;
    return this;
  }

  public Cfg setDataObjectClass(JavaClassId dataObjectClass) {
    this.dataObjectClass = dataObjectClass;
    return this;
  }

  public Cfg setMapperXmlMustache(String mapperXmlMustache) {
    this.mapperXmlMustache = mapperXmlMustache;
    return this;
  }
}
