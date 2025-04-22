package com.kantvai.kantvplayer.utils.database.builder;

import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;

import com.kantvai.kantvplayer.app.IApplication;
import com.kantvai.kantvplayer.utils.database.DataBaseInfo;
import com.kantvai.kantvplayer.utils.database.callback.QueryAsyncCallback;
import com.kantvai.kantvplayer.utils.database.callback.QueryAsyncResultCallback;
import com.kantvai.kantvplayer.utils.database.callback.QuerySyncCallback;
import com.kantvai.kantvplayer.utils.database.callback.QuerySyncResultCallback;

import java.util.ArrayList;
import java.util.List;

import io.reactivex.annotations.CheckReturnValue;

public class QueryBuilder {
    private SQLiteDatabase sqLiteDatabase;
    private int tablePosition;
    private String[] colNames;
    private List<String> whereClause;
    private List<String> whereArgs;
    private String groupColName;
    private String having = null;
    private boolean isAsc = true;
    private String orderByColName;
    private String limit = null;

    QueryBuilder(int tablePosition, SQLiteDatabase sqLiteDatabase) {
        this.tablePosition = tablePosition;
        this.sqLiteDatabase = sqLiteDatabase;
        whereClause = new ArrayList<>();
        whereArgs = new ArrayList<>();
    }

    @CheckReturnValue
    public QueryBuilder queryColumns(String... colNames) {
        this.colNames = colNames;
        return this;
    }

    @CheckReturnValue
    public QueryBuilder where(String colName, String value) {
        if (value == null) value = "";
        DataBaseInfo.checkColumnName(colName, tablePosition);
        String whereClauseText = colName + " = ?";
        whereClause.add(whereClauseText);
        whereArgs.add(value);
        return this;
    }

    @CheckReturnValue
    public QueryBuilder setGroupByColumn(String colName) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        this.groupColName = colName;
        return this;
    }

    @CheckReturnValue
    public QueryBuilder setHaving(String having) {
        this.having = having;
        return this;
    }

    @CheckReturnValue
    public QueryBuilder setOrderByColumnAsc(String colName) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        isAsc = true;
        this.orderByColName = colName;
        return this;
    }

    @CheckReturnValue
    public QueryBuilder setOrderByColumnDesc(String colName) {
        DataBaseInfo.checkColumnName(colName, tablePosition);
        isAsc = false;
        this.orderByColName = colName;
        return this;
    }

    @CheckReturnValue
    public QueryBuilder setLimit(String limit) {
        this.limit = limit;
        return this;
    }

    @CheckReturnValue
    private Cursor execute() {

        //selection
        String clause = null;
        String[] args = null;
        if (whereArgs.size() > 0) {
            args = new String[whereClause.size()];
            StringBuilder clauseBuilder = new StringBuilder();
            for (int i = 0; i < whereClause.size(); i++) {
                clauseBuilder.append(whereClause.get(i)).append(" AND ");
                args[i] = whereArgs.get(i);
            }
            if (clauseBuilder.length() > 5) {
                clause = clauseBuilder.substring(0, clauseBuilder.length() - 5);
            }
        }

        String orderByText;
        if (isAsc) {
            orderByText = orderByColName == null ? null : orderByColName + " ASC";
        } else {
            orderByText = orderByColName == null ? null : orderByColName + " DESC";
        }

        if (sqLiteDatabase.isOpen())
            return sqLiteDatabase.query(DataBaseInfo.getTableNames()[tablePosition], colNames, clause, args, groupColName, having, orderByText, limit);
        return null;
    }


    @CheckReturnValue
    public synchronized <T> T executeAsync(QuerySyncResultCallback<T> callBack) {
        ActionBuilder.checkThreadLocal();

        Cursor cursor = execute();
        T result = callBack.onQuery(cursor);
        if (cursor != null && !cursor.isClosed()) {
            cursor.close();
        }
        return result;
    }


    public synchronized void executeAsync(QuerySyncCallback callBack) {
        ActionBuilder.checkThreadLocal();

        Cursor cursor = execute();
        if (cursor != null){
            callBack.onQuery(cursor);
            //自动检查Cursor是否关闭
            if (!cursor.isClosed()) {
                cursor.close();
            }
        }
    }


    public <T> void postExecute(QueryAsyncResultCallback<T> callBack) {
        IApplication.getSqlThreadPool().execute(() -> {
            Cursor cursor = execute();
            T result = callBack.onQuery(cursor);
            if (cursor != null && !cursor.isClosed()) {
                cursor.close();
            }
            IApplication.getMainHandler().post(() -> callBack.onPrepared(result));
        });
    }


    public void postExecute(QueryAsyncCallback callBack) {
        IApplication.getSqlThreadPool().execute(() -> {
            Cursor cursor = execute();
            if (cursor != null){
                callBack.onQuery(cursor);
                if (!cursor.isClosed()) {
                    cursor.close();
                }
            }
        });
    }
}