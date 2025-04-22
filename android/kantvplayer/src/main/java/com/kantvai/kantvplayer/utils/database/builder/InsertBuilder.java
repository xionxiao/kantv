package com.kantvai.kantvplayer.utils.database.builder;

import android.content.ContentValues;
import android.database.sqlite.SQLiteDatabase;

import com.kantvai.kantvplayer.app.IApplication;
import com.kantvai.kantvplayer.utils.database.DataBaseInfo;

import io.reactivex.annotations.CheckReturnValue;


public class InsertBuilder{
    private SQLiteDatabase sqLiteDatabase;
        private int tablePosition;
        private ContentValues mValues;

    InsertBuilder(int tablePosition, SQLiteDatabase sqLiteDatabase){
        this.sqLiteDatabase = sqLiteDatabase;
        this.tablePosition = tablePosition;
        mValues = new ContentValues();
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, String value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Byte value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Short value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Integer value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Long value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Float value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Double value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, Boolean value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    @CheckReturnValue
    public InsertBuilder param(String colName, byte[] value) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        mValues.put(colName, value);
        return this;
    }

    public synchronized void executeAsync(){
        ActionBuilder.checkThreadLocal();

        if (mValues != null && sqLiteDatabase.isOpen()){
            sqLiteDatabase.insert(DataBaseInfo.getTableNames()[tablePosition], null, mValues);
        }
    }

    public void postExecute(){
        IApplication.getSqlThreadPool().execute(this::executeAsync);
    }
}