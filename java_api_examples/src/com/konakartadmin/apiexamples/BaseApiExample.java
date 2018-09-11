//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
package com.konakartadmin.apiexamples;

import java.lang.reflect.InvocationTargetException;

import com.konakart.util.KKCodeDescription;
import com.konakart.util.KKConstants;
import com.konakartadmin.app.AdminEngineConfig;
import com.konakartadmin.app.KKAdminException;
import com.konakartadmin.appif.KKAdminIf;
import com.konakartadmin.ws.KKAdminEngineMgr;

/**
 * Base API Example
 */
public class BaseApiExample
{
    /**
     * Default credentials for accessing the KonaKart Admin App
     */
    protected static String DEFAULT_USERNAME = "admin@konakart.com";

    protected static String DEFAULT_PASSWORD = "princess";

    protected static final String DEFAULT_STD_KKADMIN_ENG_CLASS = "com.konakartadmin.bl.KKAdmin";

    protected static final String DEFAULT_WS_KKADMIN_ENG_CLASS = "com.konakartadmin.ws.KKWSAdmin";

    protected static final String DEFAULT_JSON_KKADMIN_ENG_CLASS = "com.konakartadmin.json.KKJSONAdminEng";

    protected static final String DEFAULT_RMI_KKADMIN_ENG_CLASS = "com.konakartadmin.rmi.KKRMIAdminEng";

    protected static final String DEFAULT_CUSTOM_KKADMIN_ENG_CLASS = "com.konakartadmin.bl.KKCustomAdminService";

    protected static final String DEFAULT_CUSTOM_KKWSADMIN_ENG_CLASS = "com.konakartadmin.ws.KKWSCustomAdminService";

    protected static final String DEF_STORE_ID = KKConstants.KONAKART_DEFAULT_STORE_ID;

    /**
     * The Admin Engine
     */
    protected static KKAdminIf eng;

    /** StoreId - Used in Multi-Store Installations to identify the store */
    private static String storeId = KKConstants.KONAKART_DEFAULT_STORE_ID;

    /** engClassName - Name of the engine to use */
    private static String engClassName = DEFAULT_STD_KKADMIN_ENG_CLASS;

    /** POJO Custom Store Service Engine */
    protected static final String kkCustomAdminEngName = DEFAULT_CUSTOM_KKADMIN_ENG_CLASS;

    /** SOAP Custom Store Service Engine */
    protected static final String kkWSCustomAdminEngName = DEFAULT_CUSTOM_KKWSADMIN_ENG_CLASS;

    /** Engine Mode - 0 = Single Store 1 = Multi-Store Multiple DBs, 2 = Multi-Store Single DB */
    private static int engineMode = KKConstants.MODE_SINGLE_STORE;

    /** debug flag */
    private static boolean debug = false;

    /** customersShared flag */
    private static boolean customersShared = false;

    /** productsShared flag */
    private static boolean productsShared = false;

    /** categoriesShared flag */
    private static boolean categoriesShared = false;

    /** username */
    private static String username = DEFAULT_USERNAME;

    /** password */
    private static String password = DEFAULT_PASSWORD;
    
    protected static final String COMMON_USAGE = "                                     \n"
            + "   [-s storeId]       storeId (default " + DEF_STORE_ID + ")            \n"
            + "   [-m (0|1|2)]       engine Mode (default 0 = Single Store)            \n"
            + "   [-c (true|false)]  shared customers (default is false)               \n"
            + "   [-ps (true|false)] shared products (default is false)                \n"
            + "   [-cs (true|false)] shared categories (default is false)              \n"
            + "   [-e classname]     engine classname (default is " + engClassName + ")\n"
            + "   [-?]               prints this usage information                     \n"
            + "   [-d]               enable debug                                      \n";

    /**
     * The session id returned by a successful login
     */
    protected static String sessionId;

    static protected void parseArgs(String[] args, String usageIn, int minArgs)
    {
        String usage = usageIn;
        
        if (usage == null)
        {
            usage = COMMON_USAGE;
        }

        if (minArgs > 0 && (args == null || args.length < minArgs))
        {
            System.out.println("\n" + usage + "\nIllegal or insufficient arguments\n");
            System.exit(1);
            return;
        }

        if (args != null && args.length > 0)
        {
            for (int a = 0; a < args.length; a++)
            {
                if (args[a].equals("-m"))
                {
                    setEngineMode(Integer.parseInt(args[++a]));
                } else if (args[a].equals("-s"))
                {
                    setStoreId(args[++a]);
                } else if (args[a].equals("-e"))
                {
                    setEngClassName(args[++a]);
                } else if (args[a].equals("-c"))
                {
                    setCustomersShared(args[++a].equalsIgnoreCase("true"));
                } else if (args[a].equals("-ps"))
                {
                    setProductsShared(args[++a].equalsIgnoreCase("true"));
                } else if (args[a].equals("-cs"))
                {
                    setCategoriesShared(args[++a].equalsIgnoreCase("true"));
                } else if (args[a].equals("-?"))
                {
                    System.out.println("\n" + usage);
                    System.exit(1);
                    return;
                } else if (args[a].equals("-d"))
                {
                    setDebug(true);
                } else
                {
                    System.out.println("\n" + usage + "\nUnknown argument: " + args[a] + "\n");
                    System.exit(1);
                }
            }
        }

        if (isDebug())
        {
            System.out.println("store:              " + getStoreId());
            System.out.println("Engine Class:       " + getEngClassName());
            System.out.println("Engine Mode:        "
                    + KKCodeDescription.engineModeToString(getEngineMode()));
            System.out.println("Shared Customers?   " + isCustomersShared());
            System.out.println("Shared Products?    " + isProductsShared());
            System.out.println("Shared Categories?  " + isCategoriesShared());
            System.out.println();
        }
    }

    /**
     * Initialise a KonaKart Admin engine instance by name and perform a login to get a session id.
     * 
     * @throws ClassNotFoundException
     * @throws InstantiationException
     * @throws IllegalAccessException
     * @throws KKAdminException
     * @throws InvocationTargetException
     * @throws IllegalArgumentException
     */
    static protected void init(int engMode, String sId, String engineClass, boolean custShared,
            boolean prodShared, boolean catShared, String[] properties)
            throws ClassNotFoundException, InstantiationException, IllegalAccessException,
            KKAdminException, IllegalArgumentException, InvocationTargetException
    {
        /*
         * Instantiate a KonaKart Admin Engine instance by name
         */
        KKAdminEngineMgr kkAdminEngMgr = new KKAdminEngineMgr();
        AdminEngineConfig adEngConf = new AdminEngineConfig();
        adEngConf.setMode(engMode);
        adEngConf.setPropertiesFileName(KKConstants.KONAKARTADMIN_PROPERTIES_FILE);
        adEngConf.setAxisClientFileName(KKConstants.KONAKARTADMIN_WS_CLIENT_PROPERTIES_FILE);
        adEngConf.setStoreId(sId);
        adEngConf.setCustomersShared(custShared);
        adEngConf.setProductsShared(prodShared);
        adEngConf.setCategoriesShared(catShared);
        adEngConf.setProperties(properties);

        /*
         * This creates a KonaKart Admin Engine by name using the constructor that requires an
         * AdminEngineConfig object. This is the recommended approach.
         */
        setEng(kkAdminEngMgr.getKKAdminByName(engineClass, adEngConf));

        /*
         * Login with default credentials
         */
        sessionId = getEng().login(getUsername(), getPassword());
    }

    /**
     * @return the eng
     */
    public static KKAdminIf getEng()
    {
        return eng;
    }

    /**
     * @param eng
     *            the eng to set
     */
    public static void setEng(KKAdminIf eng)
    {
        BaseApiExample.eng = eng;
    }

    /**
     * @return the storeId
     */
    public static String getStoreId()
    {
        return storeId;
    }

    /**
     * @param storeId
     *            the storeId to set
     */
    public static void setStoreId(String storeId)
    {
        BaseApiExample.storeId = storeId;
    }

    /**
     * @return the engineMode
     */
    public static int getEngineMode()
    {
        return engineMode;
    }

    /**
     * @param engineMode
     *            the engineMode to set
     */
    public static void setEngineMode(int engineMode)
    {
        BaseApiExample.engineMode = engineMode;
    }

    /**
     * @return the debug
     */
    public static boolean isDebug()
    {
        return debug;
    }

    /**
     * @param debug
     *            the debug to set
     */
    public static void setDebug(boolean debug)
    {
        BaseApiExample.debug = debug;
    }

    /**
     * @return the customersShared
     */
    public static boolean isCustomersShared()
    {
        return customersShared;
    }

    /**
     * @param customersShared
     *            the customersShared to set
     */
    public static void setCustomersShared(boolean customersShared)
    {
        BaseApiExample.customersShared = customersShared;
    }

    /**
     * @return the productsShared
     */
    public static boolean isProductsShared()
    {
        return productsShared;
    }

    /**
     * @param productsShared
     *            the productsShared to set
     */
    public static void setProductsShared(boolean productsShared)
    {
        BaseApiExample.productsShared = productsShared;
    }

    /**
     * @return the sessionId
     */
    public static String getSessionId()
    {
        return sessionId;
    }

    /**
     * @param sessionId
     *            the sessionId to set
     */
    public static void setSessionId(String sessionId)
    {
        BaseApiExample.sessionId = sessionId;
    }

    /**
     * @return the engClassName
     */
    public static String getEngClassName()
    {
        return engClassName;
    }

    /**
     * @param engClassName
     *            the engClassName to set
     */
    public static void setEngClassName(String engClassName)
    {
        BaseApiExample.engClassName = engClassName;
    }

    /**
     * @return the categoriesShared
     */
    public static boolean isCategoriesShared()
    {
        return categoriesShared;
    }

    /**
     * @param categoriesShared
     *            the categoriesShared to set
     */
    public static void setCategoriesShared(boolean categoriesShared)
    {
        BaseApiExample.categoriesShared = categoriesShared;
    }

    /**
     * @return the username
     */
    public static String getUsername()
    {
        return username;
    }

    /**
     * @param username the username to set
     */
    public static void setUsername(String username)
    {
        BaseApiExample.username = username;
    }

    /**
     * @return the password
     */
    public static String getPassword()
    {
        return password;
    }

    /**
     * @param password the password to set
     */
    public static void setPassword(String password)
    {
        BaseApiExample.password = password;
    }
}
